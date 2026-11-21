import math
from collections import deque
from itertools import islice
from statistics import median_low

from nanovllm.config import Config, cuda_graph_batch_sizes
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.ngram import NgramProposer


class Scheduler:
    # Adaptive k (speculative_min_acceptance): for every draft position j = 1..K the scheduler keeps
    # the rate at which position j was accepted, over about the last ACCEPTANCE_WINDOW sequence-steps
    # that proposed it. A decode step proposes the leading positions whose rate is at least the
    # threshold (none: it runs without drafts), except one probe step with all K positions every
    # PROBE_INTERVAL steps, which is how later positions, and drafts at all, come back once the
    # workload changes.
    # With speculative_per_sequence every sequence keeps its own rates (one sample per position
    # per step) and gets its own number of drafts, so a batch can hold sequences with different
    # numbers of drafts (a ragged batch, see ModelRunner.prepare_decode); a sequence proposes every
    # position on its first step, and the probes stay the batch's, so that a probe is one uniform
    # step rather than a ragged step per sequence.
    ACCEPTANCE_WINDOW = 64    # sequence-steps per position, or ACCEPTANCE_MIN_STEPS steps if that is more
    ACCEPTANCE_MIN_STEPS = 8  # a large batch fills 64 samples in two steps, too few to decide on
    ACCEPTANCE_WINDOW_PER_SEQUENCE = 16    # steps of one sequence: small, so a sequence that stops copying its prompt is noticed within a few steps
    ACCEPTANCE_PRIOR_PER_SEQUENCE = 1      # accepted samples a new sequence starts with per position: one rejection leaves it at 1/2, not 0
    PROBE_INTERVAL = 16       # decode steps
    # Adaptive k by cost (speculative_k_by_cost): instead of a threshold, a decode step proposes
    # the number of drafts k (0 to K) with the most expected tokens per second: (1 + the acceptance
    # rates of positions 1..k) / the step time of k drafts at this batch size and context. The
    # engine times every decode step that follows a decode step (record_step_time), with the KV
    # tokens its sequences attend to (their cached tokens, summed); the scheduler keeps the last
    # STEP_TIME_WINDOW of them per batch-size bucket and k. A step reads the KV of every context,
    # so its time grows as the contexts do, and the times of different k, taken at different
    # moments, are compared through a line per k, T_k(x) = c_k + s_k * x at x KV tokens (see
    # step_time_lines), at the KV tokens of the step being scheduled. A k with fewer than
    # MIN_STEP_TIME_SAMPLES times in its bucket, counting only those of the last STEP_TIME_MAX_AGE
    # decode steps, is proposed before any choice is made, the largest first (the first steps
    # proposing K also measure every position's acceptance), so every k is timed and a k timed
    # long ago (another workload) is timed again. The probe steps stay: they measure the positions
    # above the chosen k, and their times, spread over many steps, how the time of K drafts grows.
    STEP_TIME_WINDOW = 32
    MIN_STEP_TIME_SAMPLES = 2
    STEP_TIME_MAX_AGE = 512   # decode steps
    STEP_TIME_SMALL_BATCH = 8  # batches up to this size share a bucket: a step reads the weights once, whatever the batch (A40, 8B: 25.6 ms at batch 1, about 27 at batch 8)

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.num_speculative_tokens = config.num_speculative_tokens
        self.has_draft_model = config.speculative_model is not None    # the model runner drafts on the GPU
        self.has_eagle_head = self.has_draft_model and getattr(config, "speculative_method", None) == "eagle3"    # ... with an EAGLE-3 head, whose rows sit one position behind the target's
        self.speculative_disable_by_batch_size = config.speculative_disable_by_batch_size
        self.speculative_min_acceptance = config.speculative_min_acceptance
        self.speculative_per_sequence = bool(config.speculative_per_sequence) and self.num_speculative_tokens > 0
        # Per draft position (index 1..K): its acceptance rate over about the last ACCEPTANCE_WINDOW
        # sequence-steps that proposed it, and how many of those it has seen. Trusted until measured.
        self.acceptance = [1.0] * (self.num_speculative_tokens + 1)
        self.num_acceptance_samples = [0] * (self.num_speculative_tokens + 1)
        self.num_steps_without_probe = 0    # consecutive decode steps that proposed fewer than K drafts
        self.k_by_cost = bool(config.speculative_k_by_cost) and self.num_speculative_tokens > 0
        # A separate draft model runs once per draft, reading its own KV cache each time: the KV cost
        # of a step grows with its drafts. An EAGLE-3 head's cache is one layer, n-gram drafts have none.
        self.kv_slope_by_drafts = self.has_draft_model and not self.has_eagle_head
        # Batch-size buckets for the step times: the CUDA graph sizes (a batch runs the graph of
        # the next size up, so it takes that size's time), the small ones merged.
        self.step_time_buckets = [size for size in cuda_graph_batch_sizes(config.max_num_seqs) if size >= self.STEP_TIME_SMALL_BATCH]
        self.step_times: dict[int, list[deque[tuple[int, int, float]]]] = {}    # bucket -> per k: (decode step index, KV tokens, seconds), the last STEP_TIME_WINDOW
        self.decode_step_index = 0    # decode steps scheduled so far (unlike num_decode_steps, never reset): the clock samples age by
        self.proposer = NgramProposer(config.num_speculative_tokens) if config.num_speculative_tokens and not self.has_draft_model else None
        self.reset_stats()
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if self.speculative_per_sequence:
            # Trusted until measured, like the batch-wide rates, but with ACCEPTANCE_PRIOR_PER_SEQUENCE
            # optimistic samples behind the trust: a sequence has one sample per step and position, so
            # without them its first rejected draft (the n-gram proposer's, before the prompt is
            # repeated; a draft model's near-tie) would strip it of that position until the next probe.
            seq.acceptance = [1.0] * (self.num_speculative_tokens + 1)
            seq.num_acceptance_samples = [0] + [self.ACCEPTANCE_PRIOR_PER_SEQUENCE] * self.num_speculative_tokens
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                seq.num_draft_cached_tokens = seq.num_cached_tokens    # a hashed block holds both models' KV
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        k = self.num_speculative_tokens
        capped = k and self.speculative_disable_by_batch_size is not None and len(self.running) > self.speculative_disable_by_batch_size
        if capped:
            k = 0    # a batch this large keeps the GPU busy by itself; the k extra positions would not be free
        per_sequence = k > 0 and self.speculative_per_sequence    # each sequence picks its own number of drafts below
        probe = self.num_steps_without_probe >= self.PROBE_INTERVAL    # every sequence proposes all K positions, to measure the left-out ones again
        if k and self.speculative_min_acceptance is not None and not per_sequence and not probe:
            k = self.num_acceptable_drafts()    # positions rejected too often are left out; the next probe step measures them again
        elif k and self.k_by_cost and not probe:
            batch_size = min(len(self.running), self.max_num_seqs)
            k = self.num_drafts_by_cost(batch_size, sum(seq.num_cached_tokens for seq in islice(self.running, batch_size)))
        kept_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.num_completion_tokens + seq.num_inflight >= seq.max_tokens:    # the step in flight may finish it
                kept_seqs.append(seq)
                continue
            num_drafts = (self.num_speculative_tokens if probe else self.num_acceptable_drafts(seq)) if per_sequence else k
            num_decode_tokens = 1 + num_drafts
            while not self.block_manager.can_append(seq, num_decode_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = num_decode_tokens
                seq.is_prefill = False
                if self.proposer:
                    seq.draft_token_ids = self.proposer.propose(seq)[:num_drafts] if num_drafts else []
                self.block_manager.may_append(seq, num_decode_tokens)
                scheduled_seqs.append(seq)
                kept_seqs.append(seq)
                if per_sequence:
                    self.num_steps_by_k[num_drafts] += 1    # sequence-steps in this mode
        self.running.extendleft(reversed(kept_seqs))
        if scheduled_seqs:
            self.num_decode_steps += 1
            self.decode_step_index += 1
            self.num_capped_steps += bool(capped)
            if per_sequence:
                self.num_ragged_steps += len({seq.num_scheduled_tokens for seq in scheduled_seqs}) > 1
                self.num_steps_without_probe = 0 if all(seq.num_scheduled_tokens == 1 + self.num_speculative_tokens for seq in scheduled_seqs) else self.num_steps_without_probe + 1
            else:
                self.num_steps_without_probe = 0 if k == self.num_speculative_tokens else self.num_steps_without_probe + 1
                self.num_steps_by_k[k] += 1
        return scheduled_seqs, False

    def reset_stats(self):
        """Speculative decoding statistics (cumulative, unlike the acceptance window): drafts proposed
        and accepted in all, per draft position, decode steps by their number of drafts (with
        speculative_per_sequence: sequence-steps), decode steps, and the ragged and capped ones among them."""
        k = self.num_speculative_tokens
        self.num_draft_tokens = self.num_accepted_tokens = 0
        self.num_proposed_by_position = [0] * (k + 1)
        self.num_accepted_by_position = [0] * (k + 1)
        self.num_steps_by_k = [0] * (k + 1)
        self.num_decode_steps = self.num_ragged_steps = self.num_capped_steps = 0    # capped: drafts off by speculative_disable_by_batch_size

    def num_acceptable_drafts(self, seq: Sequence | None = None) -> int:
        """The leading draft positions whose measured acceptance rate reaches the threshold: the
        batch's rates, or with a sequence its own, which admit every position on its first step
        (nothing measured yet). The probe steps are decided by the caller."""
        acceptance = self.acceptance if seq is None else seq.acceptance
        k = 0
        while k < self.num_speculative_tokens and acceptance[k + 1] >= self.speculative_min_acceptance:
            k += 1
        return k

    def step_time_bucket(self, batch_size: int) -> int:
        return next((size for size in self.step_time_buckets if size >= batch_size), batch_size)    # above the largest graph: eager, its own size

    def step_time_table(self, batch_size: int) -> list[deque[tuple[int, int, float]]]:
        bucket = self.step_time_bucket(batch_size)
        if bucket not in self.step_times:
            self.step_times[bucket] = [deque(maxlen=self.STEP_TIME_WINDOW) for _ in range(self.num_speculative_tokens + 1)]
        return self.step_times[bucket]

    def record_step_time(self, batch_size: int, num_drafts: int, kv_tokens: int, seconds: float):
        """A decode step of batch_size sequences with num_drafts drafts each, attending to kv_tokens
        cached tokens in all, took this long (see LLMEngine.time_decode_step)."""
        self.step_time_table(batch_size)[num_drafts].append((self.decode_step_index, kv_tokens, seconds))

    def step_time_lines(self, times: list[deque[tuple[int, int, float]]]) -> list[tuple[float, float]]:
        """Per k: (c_k, s_k), the step time of k drafts at x KV tokens being c_k + s_k * x seconds.
        The slope is fitted by least squares to how each k's own times moved with x: one line per
        k, with a common slope, or with a separate draft model a slope a + b * k (each draft reads
        the draft model's KV once more). A k timed at a single x tells nothing about the slope;
        when only one k moved, the common slope is fitted, and when none did, the slope is 0 (the
        times are compared as they are). A slope is kept between 0 and the steepest that leaves
        none of k's times below its line at x = 0, as the KV cannot take more than a whole step.
        c_k is the low median of (time - s_k * x) over k's times: a pause only ever adds time."""
        spread = [0.0] * len(times)    # per k: the sum of (x - mean x)^2 over its times
        moved = [0.0] * len(times)     # per k: the sum of (x - mean x) * (time - mean time)
        for k, samples in enumerate(times):
            if len(samples) > 1:
                mean_x = sum(x for _, x, _ in samples) / len(samples)
                mean_t = sum(t for _, _, t in samples) / len(samples)
                spread[k] = sum((x - mean_x) ** 2 for _, x, _ in samples)
                moved[k] = sum((x - mean_x) * (t - mean_t) for _, x, t in samples)
        s_uu, s_ku, s_kku = sum(spread), sum(k * w for k, w in enumerate(spread)), sum(k * k * w for k, w in enumerate(spread))
        m_u, m_ku = sum(moved), sum(k * m for k, m in enumerate(moved))
        # s_uu * s_kku - s_ku^2, summed without cancellation: 0 unless two k moved.
        det = sum(spread[g] * spread[h] * (g - h) ** 2 for g in range(len(times)) for h in range(g))
        a = b = 0.0
        if self.kv_slope_by_drafts and det > 0:
            a = (m_u * s_kku - m_ku * s_ku) / det
            b = (s_uu * m_ku - s_ku * m_u) / det
        elif s_uu > 0:
            a = m_u / s_uu
        lines = []
        for k, samples in enumerate(times):
            slope = min([max(0.0, a + b * k)] + [t / x for _, x, t in samples if x > 0])
            lines.append((median_low([t - slope * x for _, x, t in samples]) if samples else 0.0, slope))
        return lines

    def num_drafts_by_cost(self, batch_size: int, kv_tokens: int) -> int:
        """The number of drafts with the most expected tokens per second at this batch size and
        this many KV tokens, or the largest one still to be timed (see STEP_TIME_WINDOW). Ties go
        to fewer drafts."""
        times = self.step_time_table(batch_size)
        oldest = self.decode_step_index - self.STEP_TIME_MAX_AGE
        for samples in times:
            while samples and samples[0][0] < oldest:
                samples.popleft()
        for k in range(self.num_speculative_tokens, -1, -1):
            if len(times[k]) < self.MIN_STEP_TIME_SAMPLES:
                return k
        best_k, best_rate, expected = 0, 0.0, 1.0
        for k, (base, slope) in enumerate(self.step_time_lines(times)):
            if k:
                expected += self.acceptance[k]    # position k is accepted only with the k - 1 before it: the rates add up to the expected accepted drafts
            seconds = base + slope * kv_tokens
            rate = expected / seconds if seconds > 0 else math.inf
            if rate > best_rate:
                best_k, best_rate = k, rate
        return best_k

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_draft_cached_tokens = 0    # the prefill rebuilds both KV caches
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def commit(self, seqs: list[Sequence], is_prefill: bool) -> list[tuple[Sequence, bool, int, int, int]]:
        """Advance sequences past a step that has been launched. Its sampled tokens stay in flight
        until postprocess, so with async scheduling the next step can be scheduled first. A prefill
        chunk's KV is counted now, so that the next chunk follows it; a decode step's KV is counted
        in postprocess, once the number of accepted draft tokens is known."""
        batch = []
        for row, seq in enumerate(seqs):
            start, num_tokens = seq.num_cached_tokens, seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill:
                seq.num_cached_tokens += num_tokens
                # The draft model prefills alongside the target; an EAGLE-3 head's last row needs the
                # token after the prompt, so it lags one position once the prompt is complete.
                seq.num_draft_cached_tokens = seq.num_cached_tokens - (self.has_eagle_head and seq.num_cached_tokens == seq.num_tokens)
                num_inflight = int(seq.num_cached_tokens == seq.num_tokens)    # samples once the prompt is done
            else:
                num_inflight = num_tokens    # the newest token and the drafts: the most the step can add
            if num_inflight:
                seq.num_inflight += num_inflight
                seq.inflight_row = row
            batch.append((seq, is_prefill, start, num_tokens, num_inflight))
        return batch

    def postprocess(self, batch: list[tuple[Sequence, bool, int, int, int]], token_ids: list[int] | list[list[int]]) -> list[Sequence]:
        """token_ids has one token per sequence, or with speculative decoding, a list per decoded
        sequence: its accepted draft tokens followed by one sampled token."""
        finished = []
        # Per draft position: how many sequences proposed it this step, and how many accepted it.
        proposed = [0] * (self.num_speculative_tokens + 1)
        accepted = [0] * (self.num_speculative_tokens + 1)
        for (seq, is_prefill, start, num_tokens, num_inflight), new_token_ids in zip(batch, token_ids):
            if seq.is_finished:    # finished while this step was in flight
                continue
            stop = False
            if num_inflight:
                seq.num_inflight -= num_inflight
                speculative = isinstance(new_token_ids, list)
                num_drafts = num_tokens - 1    # the step proposed this many drafts after its input token
                for token_id in new_token_ids if speculative else [new_token_ids]:
                    seq.append_token(token_id)
                    stop = (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
                    if stop:
                        break
                if not is_prefill:
                    # The step wrote the KV of its input token and of the drafts after it; only the
                    # input token's and the accepted drafts' is valid, later steps overwrite the rest.
                    start = seq.num_cached_tokens
                    num_tokens = seq.num_tokens - start - 1
                    seq.num_cached_tokens = start + num_tokens
                    if num_drafts or not self.has_draft_model:
                        # A step with drafts first caught the draft model's KV cache up to the newest
                        # token (ModelRunner.propose), then scored every draft but the last one, so when
                        # all of them were accepted the cache lacks the newest cached token. A step
                        # without drafts (see schedule), or a sequence without drafts in a ragged step,
                        # skips the draft model: its cache falls behind until the next step with drafts,
                        # and the blocks it lacks must not be hashed yet, or a prefix cache hit would
                        # read a stale draft KV.
                        # An EAGLE-3 head scores the accepted tokens again next step, with the
                        # target's hidden states: its cache ends where it did before this step.
                        lag = self.has_draft_model and len(new_token_ids) == num_drafts + 1    # every draft accepted
                        seq.num_draft_cached_tokens = start if self.has_eagle_head else seq.num_cached_tokens - lag
                if speculative:
                    for j in range(1, num_drafts + 1):
                        proposed[j] += 1
                        accepted[j] += j < len(new_token_ids)
                    if self.speculative_per_sequence and num_drafts:
                        # The sequence's own running mean per position, over its last ACCEPTANCE_WINDOW_PER_SEQUENCE steps that proposed it.
                        window = self.ACCEPTANCE_WINDOW_PER_SEQUENCE
                        for j in range(1, num_drafts + 1):
                            seen = min(seq.num_acceptance_samples[j], window)
                            seq.acceptance[j] = (seq.acceptance[j] * seen + (j < len(new_token_ids))) / (seen + 1)
                            seq.num_acceptance_samples[j] = min(seen + 1, window)
            if seq.block_table:    # empty if preempted while this step was in flight
                # Blocks both KV caches hold completely; the ones a lagging draft cache left for later are hashed
                # once it catches up (hash_blocks scans back to them).
                valid = min(start + num_tokens, seq.num_draft_cached_tokens) if is_prefill else min(seq.num_cached_tokens, seq.num_draft_cached_tokens)
                self.block_manager.hash_blocks(seq, start, valid - start)
            if stop:
                if seq.status == SequenceStatus.RUNNING:
                    self.running.remove(seq)
                else:
                    self.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if self.proposer:
                    self.proposer.forget(seq)
                finished.append(seq)
        self.num_draft_tokens += sum(proposed)
        self.num_accepted_tokens += sum(accepted)
        for j in range(1, self.num_speculative_tokens + 1):
            if proposed[j]:
                self.num_proposed_by_position[j] += proposed[j]
                self.num_accepted_by_position[j] += accepted[j]
                # A running mean over about the last `window` samples: a plain mean until that many
                # were seen, then each step's samples replace that share of the window.
                window = max(self.ACCEPTANCE_WINDOW, self.ACCEPTANCE_MIN_STEPS * proposed[j])
                seen = min(self.num_acceptance_samples[j], window)
                self.acceptance[j] = (self.acceptance[j] * seen + accepted[j]) / (seen + proposed[j])
                self.num_acceptance_samples[j] = min(seen + proposed[j], window)
        return finished
