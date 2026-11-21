"""Adaptive k by cost (speculative_k_by_cost): a decode step proposes the number of drafts k with
the most expected tokens per second, (1 + the acceptance rates of positions 1..k) / the step time
of k drafts at this batch size and context, after every k has been timed in the batch's bucket (the
largest first). Step times taken at different context lengths are compared through a line per k in
the KV tokens the step reads. The scheduler is driven directly, with made-up sampler outputs and
step times; the engine's timing of real steps is checked by the engine simulation
(tests/engine_sim.py, the *_cost modes)."""
import types

import pytest
from engine_sim import make_config

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

K = 4
# Chat at batch 32 on the A40 with the EAGLE-3 head (jobs 1733657 and 1734939): the acceptance rate
# per position and the step time of k = 0, 2, 3, 4 drafts in ms (k = 1 was not measured: made up).
HEAD_ACCEPTANCE = [0.62, 0.37, 0.20, 0.10]
HEAD_STEP_MS = [29.1, 35.0, 39.5, 41.2, 48.6]


def make_scheduler(num_seqs, max_num_seqs=16, disable_by=None, probe=100, min_samples=2, window=8, max_age=512, eagle=False, prompt_lens=None):
    Sequence.block_size = 4
    scheduler = Scheduler(make_config(num_blocks=1024, block_size=4, max_num_seqs=max_num_seqs, max_num_batched_tokens=10 ** 6,
                                      num_speculative_tokens=K, draft_model=not eagle, eagle=eagle, speculative_disable_by_batch_size=disable_by,
                                      speculative_k_by_cost=True))
    scheduler.PROBE_INTERVAL, scheduler.MIN_STEP_TIME_SAMPLES, scheduler.STEP_TIME_WINDOW, scheduler.STEP_TIME_MAX_AGE = probe, min_samples, window, max_age
    scheduler.ACCEPTANCE_MIN_STEPS = 1
    for i in range(num_seqs):
        scheduler.add(Sequence(list(range(1, 1 + (prompt_lens[i] if prompt_lens else 3))), SamplingParams(max_tokens=10 ** 6, ignore_eos=True)))
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == num_seqs
    scheduler.postprocess(scheduler.commit(seqs, True), [5] * num_seqs)
    return scheduler


def decode(scheduler, accepted, step_ms=None):
    """One decode step: every sequence accepts `accepted` drafts (capped at the step's number), and
    the step takes step_ms[k] (as LLMEngine.time_decode_step reports it). Returns k."""
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and seqs
    k = seqs[0].num_scheduled_tokens - 1
    assert all(seq.num_scheduled_tokens == k + 1 for seq in seqs)
    batch = scheduler.commit(seqs, False)
    scheduler.postprocess(batch, [[7] * (min(accepted, k) + 1) if k else 7 for _ in seqs])
    if step_ms is not None:
        scheduler.record_step_time(len(seqs), k, sum(start for _, _, start, _, _ in batch), step_ms[k] / 1000)
    return k


def fill(scheduler, batch_size, step_ms, samples=2, kv_tokens=0):
    for k, ms in enumerate(step_ms):
        for _ in range(samples):
            scheduler.record_step_time(batch_size, k, kv_tokens, ms / 1000)


def test_every_k_is_timed_first_from_the_largest_down():
    scheduler = make_scheduler(1)
    ks = [decode(scheduler, 4, [10, 11, 12, 13, 14]) for _ in range(12)]
    # Two steps per k (MIN_STEP_TIME_SAMPLES), K first: its steps also measure every position.
    # Then every draft accepted: 5 tokens in 14 ms beats 1 in 10.
    assert ks == [4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 4, 4]
    assert scheduler.acceptance[1:] == [1.0] * K


def test_the_head_on_chat_proposes_three_drafts_where_the_threshold_proposed_two():
    # Expected tokens per ms: 1/29.1, 1.62/35.0, 1.99/39.5, 2.19/41.2 (best), 2.29/48.6. The A40
    # measured k = 3 fastest (1418 tok/s against 1397 for k = 2 and 1317 for k = 4); the threshold
    # 0.3 settled on 2.
    scheduler = make_scheduler(16)
    scheduler.acceptance[1:] = HEAD_ACCEPTANCE
    fill(scheduler, 16, HEAD_STEP_MS)
    assert scheduler.num_drafts_by_cost(16, 0) == 3
    assert decode(scheduler, 1) == 3


def test_drafts_that_are_mostly_accepted_are_all_proposed():
    # Code editing at batch 1 with the 0.6B draft model: positions accepted 0.92 / 0.86 / 0.83 / 0.80
    # of the time (job 1733657), a step 25.6 ms plus 3.3 ms per draft (13.3 ms for four, at batch 1).
    scheduler = make_scheduler(1)
    scheduler.acceptance[1:] = [0.92, 0.86, 0.83, 0.80]
    fill(scheduler, 1, [25.6, 28.9, 32.2, 35.6, 38.9])
    assert scheduler.num_drafts_by_cost(1, 0) == 4


def test_a_draft_pays_when_it_is_accepted_more_often_than_it_slows_the_step():
    # The next draft adds its acceptance rate to the tokens of a step and its time to the step's:
    # with every position accepted 5% of the time it pays for a step 4% longer, not 6.7%.
    scheduler = make_scheduler(4)
    scheduler.acceptance[1:] = [0.05, 0.0, 0.0, 0.0]
    fill(scheduler, 4, [30.0, 32.0, 33.0, 34.0, 35.0])
    assert scheduler.num_drafts_by_cost(4, 0) == 0
    fill(scheduler, 4, [30.0, 31.2, 33.0, 34.0, 35.0], samples=8)
    assert scheduler.num_drafts_by_cost(4, 0) == 1


def test_ties_go_to_fewer_drafts():
    scheduler = make_scheduler(1)
    scheduler.acceptance[1:] = [0.5, 0.0, 0.0, 0.0]
    fill(scheduler, 1, [10.0, 15.0, 15.0, 15.0, 15.0])    # 1/10 = 1.5/15
    assert scheduler.num_drafts_by_cost(1, 0) == 0
    fill(scheduler, 1, [10.0, 14.0, 14.0, 14.0, 14.0], samples=8)    # k = 1 now wins; positions 2-4 add nothing
    assert scheduler.num_drafts_by_cost(1, 0) == 1


def test_a_pause_does_not_count():
    # The low median: a step held up now and then (a garbage collection, a page fault) does not make
    # its k look slow; a k that really got slower does.
    scheduler = make_scheduler(1)
    scheduler.acceptance[1:] = [0.5] * K
    fill(scheduler, 1, [10, 30, 30, 30, 11])    # k = 4: 3 tokens in 11 ms, against 1 in 10 for k = 0
    scheduler.record_step_time(1, 4, 0, 0.5)
    scheduler.record_step_time(1, 4, 0, 0.5)
    assert scheduler.num_drafts_by_cost(1, 0) == 4    # 11, 11, 500, 500: 11
    scheduler.record_step_time(1, 4, 0, 0.5)
    assert scheduler.num_drafts_by_cost(1, 0) == 0    # 11, 11, 500, 500, 500: 500


def test_the_window_keeps_the_last_times():
    scheduler = make_scheduler(1, window=3)
    scheduler.acceptance[1:] = [0.5] * K
    fill(scheduler, 1, [10, 30, 30, 30, 50], samples=3)    # k = 4: 3 tokens in 50 ms
    assert scheduler.num_drafts_by_cost(1, 0) == 0
    fill(scheduler, 1, [10, 30, 30, 30, 11])    # two of the three k = 4 times are 11 ms now
    assert [t for _, _, t in scheduler.step_times[8][4]] == [0.05, 0.011, 0.011]
    assert scheduler.num_drafts_by_cost(1, 0) == 4


def test_old_times_are_taken_again():
    # Step times change as the contexts grow or the workload changes: a k whose times are all older
    # than STEP_TIME_MAX_AGE decode steps is proposed again, MIN_STEP_TIME_SAMPLES times.
    scheduler = make_scheduler(1, min_samples=1, max_age=6)
    ks = [decode(scheduler, 4, [10, 30, 30, 30, 11]) for _ in range(16)]
    assert ks == [4, 3, 2, 1, 0, 4, 4, 4, 4, 3, 2, 1, 0, 4, 4, 4]


def test_batch_sizes_share_a_bucket_by_cuda_graph():
    scheduler = make_scheduler(1, max_num_seqs=600)
    assert [scheduler.step_time_bucket(b) for b in (1, 8, 9, 16, 17, 32, 33, 500, 512, 513)] == [8, 8, 16, 16, 32, 32, 48, 512, 512, 513]
    scheduler.acceptance[1:] = HEAD_ACCEPTANCE
    fill(scheduler, 3, HEAD_STEP_MS)
    assert list(scheduler.step_times) == [8]
    assert scheduler.num_drafts_by_cost(8, 0) == 3    # the same bucket
    assert scheduler.num_drafts_by_cost(9, 0) == 4    # another bucket: nothing timed yet, K first
    assert list(scheduler.step_times) == [8, 16]


def test_probes_still_measure_the_positions_above_the_choice():
    scheduler = make_scheduler(1, probe=3)
    scheduler.acceptance[1:] = [1.0, 1.0, 0.0, 0.0]
    step_ms = [10, 12, 14, 16, 18]    # tokens per ms: 1/10, 2/12, 3/14 (best), 3/16, 3/18
    fill(scheduler, 1, step_ms)
    assert [decode(scheduler, 2, step_ms) for _ in range(8)] == [2, 2, 2, K, 2, 2, 2, K]


def test_batch_size_rule_still_applies():
    scheduler = make_scheduler(2, disable_by=1)
    assert [decode(scheduler, 4, [10, 11, 12, 13, 14]) for _ in range(4)] == [0] * 4    # nothing timed, but the batch is too large
    assert [len(samples) for samples in scheduler.step_times.get(8, [[]] * (K + 1))] == [4, 0, 0, 0, 0]


def test_every_query_length_has_a_cuda_graph():
    runner = ModelRunner.__new__(ModelRunner)
    runner.num_speculative_tokens = K
    runner.config = types.SimpleNamespace(speculative_disable_by_batch_size=None, speculative_min_acceptance=None, speculative_k_by_cost=True)
    assert runner.decode_query_lengths() == [1, 2, 3, 4, 5]
    runner.config.speculative_k_by_cost = False
    assert runner.decode_query_lengths() == [5]


def test_the_engine_times_decode_steps_that_follow_a_decode_step():
    from nanovllm.engine.llm_engine import LLMEngine
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = make_scheduler(1)
    engine.last_decode_time = None
    times = iter([1.0, 1.5, 2.25, 3.0, 3.5, 4.0])
    engine.clock = lambda: next(times)
    recorded = []
    engine.scheduler.record_step_time = lambda *args: recorded.append(args)
    seq = Sequence([1], SamplingParams())
    decode2, decode3, prefill = [(seq, False, 40, 3, 3), (seq, False, 2, 3, 3)], [(seq, False, 50, 4, 4), (seq, False, 7, 4, 4)], [(seq, True, 0, 9, 1)]
    for batch in (decode2, decode3, prefill, decode2, decode3, decode2):
        engine.time_decode_step(batch)
    # The first decode step and the one after the prefill have nothing to be timed from. A step
    # attends to its sequences' cached tokens (where it starts writing).
    assert recorded == [(2, 3, 57, 0.5), (2, 3, 57, 0.5), (2, 2, 42, 0.5)]


# A step reads every context's KV: on the A40 with Qwen3-8B, 144 KB per token, about 0.25 us at
# its bandwidth (job 1748113 timed chat at batch 32 without drafts at 33.0 ms early on and 35.9 ms
# over the whole run, as the contexts grew).
US = 1e-6
TARGET_SLOPE = 0.25 * US    # s per KV token
HEAD_BASE_MS = [ms - 5.0 for ms in HEAD_STEP_MS]    # HEAD_STEP_MS is at 20000 KV tokens (32 sequences of about 600)


def head_ms(k, kv_tokens):
    return HEAD_BASE_MS[k] + 1000 * TARGET_SLOPE * kv_tokens


def test_times_taken_at_other_context_lengths_are_moved_along_the_line():
    # The head on chat: k = 2 timed early, at 4000 KV tokens, k = 3 at long contexts. As they are,
    # the k = 2 times look 11 ms shorter than k = 3's (1.99 tokens in 35.5 ms against 2.19 in 46.2)
    # and k = 2 would win; at the 20000 KV tokens of the step being scheduled, k = 3 does.
    scheduler = make_scheduler(16, eagle=True)
    assert not scheduler.kv_slope_by_drafts
    scheduler.acceptance[1:] = HEAD_ACCEPTANCE
    for k in (0, 1, 2, 4):
        for _ in range(2):
            scheduler.record_step_time(16, k, 4000, head_ms(k, 4000) / 1000)
    for kv_tokens in (38000, 40000, 42000, 44000):
        scheduler.record_step_time(16, 3, kv_tokens, head_ms(3, kv_tokens) / 1000)
    lines = scheduler.step_time_lines(scheduler.step_times[16])
    assert [base * 1000 for base, _ in lines] == pytest.approx(HEAD_BASE_MS)
    assert [slope for _, slope in lines] == pytest.approx([TARGET_SLOPE] * (K + 1))
    assert scheduler.num_drafts_by_cost(16, 20000) == 3
    assert scheduler.num_drafts_by_cost(16, 4000) == 3    # the best at any context here: the shared slope adds the same to every k
    scheduler.step_times[16][3].clear()
    for kv_tokens in (38000, 40000, 42000, 44000):
        scheduler.record_step_time(16, 3, kv_tokens, head_ms(3, 44000) / 1000)    # times that did not move: no slope
    assert [slope for _, slope in scheduler.step_time_lines(scheduler.step_times[16])] == [0.0] * (K + 1)
    assert scheduler.num_drafts_by_cost(16, 20000) == 2    # compared as they are


# The 0.6B draft model reads its own KV once per draft: 112 KB per token, about 0.2 us more per
# token and draft.
DRAFT_SLOPE = 0.2 * US
DRAFT_ACCEPTANCE = [0.8, 0.7, 0.6, 0.5]


def draft_ms(k, kv_tokens):
    return 25.6 + 3.3 * k + 1000 * (TARGET_SLOPE + DRAFT_SLOPE * k) * kv_tokens


def test_a_draft_model_s_drafts_cost_more_as_the_contexts_grow():
    # Two k moved (K by its probes, the chosen k), so the slope's growth per draft is fitted. Then
    # at short contexts all four drafts pay (3.6 tokens in 38.8 ms), at 200000 KV tokens two do
    # (2.5 in 162.2 ms against 3.6 in 248.8).
    scheduler = make_scheduler(1, max_num_seqs=32)
    assert scheduler.kv_slope_by_drafts
    scheduler.acceptance[1:] = DRAFT_ACCEPTANCE
    for k, points in ((0, [20000] * 2), (1, [25000] * 2), (2, [150000, 160000, 170000]), (3, [5000] * 2), (4, [10000, 60000, 110000])):
        for kv_tokens in points:
            scheduler.record_step_time(32, k, kv_tokens, draft_ms(k, kv_tokens) / 1000)
    lines = scheduler.step_time_lines(scheduler.step_times[32])
    assert [base * 1000 for base, _ in lines] == pytest.approx([25.6 + 3.3 * k for k in range(K + 1)])
    assert [slope for _, slope in lines] == pytest.approx([TARGET_SLOPE + DRAFT_SLOPE * k for k in range(K + 1)])
    assert scheduler.num_drafts_by_cost(32, 1000) == 4
    assert scheduler.num_drafts_by_cost(32, 200000) == 2
    # With one k moved, its slope is every k's: here k = 2's, 0.65 us per token.
    for k in (2, 4):
        scheduler.step_times[32][k].clear()
    for kv_tokens in (150000, 160000, 170000):
        scheduler.record_step_time(32, 2, kv_tokens, draft_ms(2, kv_tokens) / 1000)
    for _ in range(2):
        scheduler.record_step_time(32, 4, 10000, draft_ms(4, 10000) / 1000)
    assert [slope for _, slope in scheduler.step_time_lines(scheduler.step_times[32])] == pytest.approx([TARGET_SLOPE + 2 * DRAFT_SLOPE] * (K + 1))


def test_without_a_draft_model_the_slope_is_common():
    # The head on chat, k = 3 chosen over four steps and k = 4 probed over many: their times grew
    # by 0.25 and 0.27 us per KV token. With the head one slope serves every k, their deviations
    # pooled; the same times from a draft model give a slope that grows by 0.02 us per draft.
    def times(scheduler):
        for k in (0, 1, 2):
            for _ in range(2):
                scheduler.record_step_time(16, k, 4000, head_ms(k, 4000) / 1000)
        for kv_tokens in (38000, 40000, 42000, 44000):
            scheduler.record_step_time(16, 3, kv_tokens, head_ms(3, kv_tokens) / 1000)
        for kv_tokens in (10000, 30000, 50000):
            scheduler.record_step_time(16, 4, kv_tokens, (HEAD_BASE_MS[4] + 0.27e-3 * kv_tokens) / 1000)
        return [slope / US for _, slope in scheduler.step_time_lines(scheduler.step_times[16])]

    pooled = (2e7 * 0.25 + 8e8 * 0.27) / (2e7 + 8e8)    # k = 3's deviations from 41000 tokens, k = 4's from 30000
    assert times(make_scheduler(16, eagle=True)) == pytest.approx([pooled] * (K + 1))
    assert times(make_scheduler(16)) == pytest.approx([0.19 + 0.02 * k for k in range(K + 1)])


def test_a_slope_stays_between_none_and_the_whole_step():
    scheduler = make_scheduler(1)
    # Times that fell as the contexts grew (another workload, a pause early on): no slope.
    scheduler.record_step_time(1, 0, 1000, 0.012)
    scheduler.record_step_time(1, 0, 2000, 0.010)
    assert scheduler.step_time_lines(scheduler.step_times[8])[0] == (0.010, 0.0)
    # Times that grew faster than the KV can explain: 10 ms at 1000 tokens, 30 at 2000 would put
    # -10 ms at none; the slope stops where the 1000-token step would be all KV.
    scheduler.step_times[8][0].clear()
    scheduler.record_step_time(1, 0, 1000, 0.010)
    scheduler.record_step_time(1, 0, 2000, 0.030)
    base, slope = scheduler.step_time_lines(scheduler.step_times[8])[0]
    assert slope == pytest.approx(0.010 / 1000) and base == pytest.approx(0.0, abs=1e-12)


def test_the_choice_is_made_at_the_kv_tokens_of_the_batch_being_scheduled():
    # The sequences the step will run, the first max_num_seqs running, with their cached tokens.
    scheduler = make_scheduler(3, prompt_lens=[3, 5, 9])
    scheduler.max_num_seqs = 2
    asked = []
    scheduler.num_drafts_by_cost = lambda batch_size, kv_tokens: asked.append((batch_size, kv_tokens)) or 1
    seqs, _ = scheduler.schedule()
    assert asked == [(2, 3 + 5)] and [seq.num_scheduled_tokens for seq in seqs] == [2, 2]
