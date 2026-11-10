"""Randomized engine simulation: the real LLMEngine.step, Scheduler, BlockManager, Sequence and
ModelRunner.prepare_prefill/prepare_decode (plus ModelRunner.propose with a draft model), driven by
a fake model instead of a GPU, with and without async scheduling, and with speculative decoding
(n-gram or fake draft model proposals checked by the real RejectionSampler).

For every position p of a sequence, the fake model writes prefix_hash(tokens[:p+1]) into a paged
KV cache at the slot that slot_mapping gives, then "attends" by reading every context position
through block_tables and checks each value it reads. The next token is a function of the prefix,
so each request has a reference output that does not depend on batching, chunked prefill, prefix
caching, preemption or speculation, and the engine must reproduce it exactly. A fake draft model
does the same with its own KV cache and hash seed; its proposals only affect the acceptance rate,
so the checks on its cache reads are what catch a stale draft KV. A fake EAGLE-3 head does the
same one position behind the target, reading the target's hidden states (the prefix hashes the
fake target writes to a hidden cache) at its catch-up rows and its own output at its draft rows.
"""
import collections
import hashlib
import pickle
import random
import types

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import DraftSampler, RejectionSampler, accepted_token_ids
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, set_context, reset_context

VOCAB = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    # where ModelRunner's tensors live
EOS = 0
TARGET_SEED = 12345
DRAFT_SEED = 54321


class Violation(AssertionError):
    pass


def make_config(num_blocks, block_size, max_num_seqs, max_num_batched_tokens, eos=EOS, num_speculative_tokens=0, draft_model=False,
                speculative_disable_by_batch_size=None, speculative_min_acceptance=None, speculative_per_sequence=False, speculative_ragged="pad", eagle=False):
    # Config itself requires a model directory and block sizes that are multiples of 256.
    return types.SimpleNamespace(num_kvcache_blocks=num_blocks, kvcache_block_size=block_size,
                                 max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens, eos=eos,
                                 num_speculative_tokens=num_speculative_tokens, speculative_model="fake" if draft_model or eagle else None,
                                 speculative_method="eagle3" if eagle else "model" if draft_model else None,
                                 speculative_disable_by_batch_size=speculative_disable_by_batch_size, speculative_min_acceptance=speculative_min_acceptance,
                                 speculative_per_sequence=speculative_per_sequence, speculative_ragged=speculative_ragged)


def make_runner(block_size, world_size=1, rank=0, num_speculative_tokens=0):
    # Only the attributes that prepare_prefill/prepare_decode/propose read; no model or KV cache.
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = block_size
    runner.world_size = world_size
    runner.rank = rank
    runner.enforce_eager = True
    runner.num_speculative_tokens = num_speculative_tokens
    runner.draft_model = None
    runner.eagle = False
    runner.draft_inputs = None
    runner.draft_layout = runner.sampler_layout = None
    runner.ragged_mode = "varlen"
    runner.sampled_drafts = True    # a draft model samples its drafts and keeps their logits (see propose)
    runner.draft_logits = runner.draft_log_norms = None
    runner.draft_sampler = DraftSampler()
    runner.graphs = runner.draft_graphs = runner.graph_pool = None    # no CUDA graphs: forward runs eagerly
    return runner


def make_engine(num_blocks, block_size, max_num_seqs, max_num_batched_tokens, async_scheduling=False, num_speculative_tokens=0, draft_model=False,
                speculative_disable_by_batch_size=None, speculative_min_acceptance=None, speculative_per_sequence=False, speculative_ragged="pad",
                speculative_sampled_drafts=True, eagle=False):
    assert not (async_scheduling and num_speculative_tokens and not (draft_model or eagle))
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = Scheduler(make_config(num_blocks, block_size, max_num_seqs, max_num_batched_tokens,
                                             num_speculative_tokens=num_speculative_tokens, draft_model=draft_model,
                                             speculative_disable_by_batch_size=speculative_disable_by_batch_size,
                                             speculative_min_acceptance=speculative_min_acceptance, speculative_per_sequence=speculative_per_sequence, eagle=eagle))
    engine.model_runner = FakeModel(engine.scheduler, block_size, num_blocks, num_speculative_tokens=num_speculative_tokens, draft_model=draft_model, eagle=eagle)
    engine.model_runner.runner.ragged_mode = speculative_ragged
    engine.model_runner.runner.sampled_drafts = speculative_sampled_drafts
    engine.async_scheduling = async_scheduling
    engine.pending = None
    return engine


def prefix_hashes(token_ids, h=TARGET_SEED):
    out = []
    for t in token_ids:
        h = int.from_bytes(hashlib.blake2b(f"{h}:{t}".encode(), digest_size=8).digest(), "little")
        out.append(h)
    return out


def fits_int64(h):
    """A prefix hash cut to 62 bits, as the fake hidden cache (a torch.int64 tensor) stores it."""
    return h & ((1 << 62) - 1)


def next_token(token_ids, h):
    """The fake model's next token after token_ids, whose prefix hash is h. Prompts that start with
    an even token get mostly periodic outputs, which n-gram drafts often predict; the others get
    pseudo-random outputs, whose drafts are mostly rejected."""
    r = h >> 7
    period = 1 + token_ids[0] // 2 % 3
    if token_ids[0] % 2 == 0 and r % 8 and len(token_ids) >= period:
        return token_ids[-period]
    return r % VOCAB


def draft_next_token(token_ids, h, draft_h, miss_mod=4):
    """The fake draft model's guess after token_ids: the target's next token, except one time in
    miss_mod (miss_mod=1: never the target's, so the drafts are rejected)."""
    return next_token(token_ids, h) if (draft_h >> 9) % miss_mod else draft_h % VOCAB


def reference(prompt, sp: SamplingParams):
    token_ids = list(prompt)
    h = prefix_hashes(token_ids)[-1]
    out = []
    while True:
        t = next_token(token_ids, h)
        token_ids.append(t)
        h = prefix_hashes([t], h)[0]
        out.append(t)
        if (not sp.ignore_eos and t == EOS) or len(out) == sp.max_tokens:
            return out


def prepare(runner, seqs, is_prefill):
    input_ids, positions = runner.prepare_prefill(seqs) if is_prefill else runner.prepare_decode(seqs)
    c = get_context()
    for name in ("slot_mapping", "context_lens", "cu_seqlens_q", "cu_seqlens_k"):    # the attention kernels want int32
        t = getattr(c, name)
        if t is not None and t.dtype != torch.int32:
            raise Violation(f"{name} has dtype {t.dtype}, not int32")
    meta = dict(
        input_ids=input_ids.tolist(),
        positions=positions.tolist(),
        slot_mapping=c.slot_mapping.tolist(),
        cu_seqlens_q=c.cu_seqlens_q.tolist() if c.cu_seqlens_q is not None else None,
        cu_seqlens_k=c.cu_seqlens_k.tolist() if c.cu_seqlens_k is not None else None,
        max_seqlen_q=c.max_seqlen_q,
        max_seqlen_k=c.max_seqlen_k,
        context_lens=c.context_lens.tolist() if c.context_lens is not None else None,
        block_tables=c.block_tables.tolist() if c.block_tables is not None else None,
        draft_inputs=[t.tolist() for t in runner.draft_inputs] if not is_prefill and runner.draft_inputs is not None else None,
        draft_dtypes=[t.dtype for t in runner.draft_inputs] if not is_prefill and runner.draft_inputs is not None else None,
        draft_layout=[t.tolist() if t is not None else None for t in runner.draft_layout] if not is_prefill and runner.draft_layout is not None else None,
        sampler_layout=[t.tolist() for t in runner.sampler_layout] if not is_prefill and runner.sampler_layout is not None else None,
    )
    reset_context()
    return meta


class FakeOutput:

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def result(self):
        return self.token_ids


class FakeDraftModel:
    """Stands in for the draft Qwen3ForCausalLM inside the real ModelRunner.propose: a callable
    that reads the context like the attention layers do, keeps the draft KV cache, and whose
    "hidden states" are the draft's next tokens (compute_logits turns them into one-hot logits)."""

    def __init__(self, model: "FakeModel"):
        self.model = model
        self.seqs = []
        self.token_ids = []     # per sequence: its tokens plus the drafts proposed so far
        self.drafts = []
        self.num_drafts = []    # per sequence: how many drafts this step proposes for it

    def begin(self, seqs, token_ids, num_drafts):
        """token_ids: per sequence, its tokens including those still in flight."""
        self.seqs = seqs
        self.token_ids = [list(t) for t in token_ids]
        self.drafts = [[] for _ in seqs]
        self.num_drafts = list(num_drafts)

    def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bs = self.model.block_size
        c = get_context()
        if c.is_prefill:
            raise Violation("the draft model was called for prefill through propose")
        if c.slot_mapping.dtype != torch.int32 or c.context_lens.dtype != torch.int32:
            raise Violation(f"draft step slot_mapping {c.slot_mapping.dtype} context_lens {c.context_lens.dtype}, not int32")
        ids, pos, slots = input_ids.tolist(), positions.tolist(), c.slot_mapping.tolist()
        context_lens, block_tables = c.context_lens.tolist(), c.block_tables.tolist()
        q = len(ids) // len(self.seqs)
        if len(ids) != q * len(self.seqs) or len(pos) != len(ids) or len(slots) != len(ids):
            raise Violation(f"draft step inputs: {len(ids)} ids, {len(pos)} positions, {len(slots)} slots for {len(self.seqs)} seqs")
        outputs = []
        step = max(map(len, self.drafts))    # draft steps run so far
        for i, seq in enumerate(self.seqs):
            token_ids, drafts = self.token_ids[i], self.drafts[i]
            n = len(token_ids)
            step_ids, step_pos, step_slots = ids[i * q:(i + 1) * q], pos[i * q:(i + 1) * q], slots[i * q:(i + 1) * q]
            if len(drafts) >= self.num_drafts[i]:
                # A sequence that has all its drafts (none, or fewer than the batch's most) rides along
                # as a dummy row: its KV masked, its context length its own, its proposal dropped.
                if any(s != -1 for s in step_slots):
                    raise Violation(f"seq {seq.seq_id}: draft step {step} writes slots {step_slots} after its {self.num_drafts[i]} drafts")
                if context_lens[i] != n or any(p >= n for p in step_pos):
                    raise Violation(f"seq {seq.seq_id}: dummy draft step {step} positions {step_pos} context_len {context_lens[i]} for {n} tokens")
                outputs.extend([-1] * (q - 1) + [0])
                continue
            if drafts:    # one token per sequence: the newest draft
                expected_ids, expected_pos = [drafts[-1]], [n - 1]
            else:
                # The catch-up step scores the last q tokens; a shorter sequence is padded in front. The
                # padded rows and the ones already cached have a masked slot, and any id passes there.
                expected_pos = [max(0, p) for p in range(n - q, n)]
                expected_ids = [token_ids[p] if s != -1 else step_ids[j] for j, (p, s) in enumerate(zip(range(n - q, n), step_slots))]
                if step_slots[-1] == -1:
                    raise Violation(f"seq {seq.seq_id}: the catch-up step does not write the newest token's draft KV")
            if step_ids != expected_ids:
                raise Violation(f"seq {seq.seq_id}: draft step {len(drafts)} inputs {step_ids} != {expected_ids}")
            if step_pos != expected_pos or context_lens[i] != n:
                raise Violation(f"seq {seq.seq_id}: draft step {len(drafts)} positions {step_pos} context_len {context_lens[i]} for {n} tokens")
            truth = prefix_hashes(token_ids, DRAFT_SEED)
            for j in range(q):
                p = n - q + j
                if step_slots[j] == -1:
                    continue    # already cached, or padding
                if p < 0:
                    raise Violation(f"seq {seq.seq_id}: a padding row of the catch-up step writes slot {step_slots[j]}")
                self.model._write(self.model.draft_kv, seq, p, step_slots[j], truth[p])
            for p in range(n):
                if self.model.draft_kv[block_tables[i][p // bs] * bs + p % bs] != truth[p]:
                    raise Violation(f"seq {seq.seq_id}: draft step {len(drafts)} read wrong draft KV at position {p} (context_len {n})")
            draft = draft_next_token(token_ids, prefix_hashes(token_ids)[-1], truth[-1], self.model.draft_miss_mod(token_ids, len(drafts)))
            drafts.append(draft)
            token_ids.append(draft)
            outputs.extend([-1] * (q - 1) + [draft])    # only the last row of each sequence is scored
        return torch.tensor(outputs, dtype=torch.int64).unsqueeze(1)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        tokens = hidden_states[:, 0]
        if (tokens < 0).any():
            raise Violation("propose scored a draft step row other than the last one of a sequence")
        logits = torch.full((tokens.numel(), VOCAB), -1000.0)
        logits[torch.arange(tokens.numel()), tokens] = 0.0
        return logits


class FakeEagleHead:
    """Stands in for Eagle3Head inside the real ModelRunner.propose. Its row at position p takes
    the target's hidden state at p (read from the hidden cache, where the fake target wrote the
    prefix hash of tokens[:p+1]) and token p + 1, so its catch-up rows end one position before
    the newest token, and its context is one shorter than the target's; a draft row takes the
    head's own output of the step before. Its KV at position p is (the draft hash of tokens[:p+1],
    token p + 1): in a block shared through the prefix cache, the last position's row was computed
    by whichever sequence hashed the block, with its own next token (the block's hash covers the
    tokens up to it only), so only the hash part is checked there (FakeModel.head_kv_matches)."""

    def __init__(self, model: "FakeModel"):
        self.model = model
        self.hidden_cache = model.hidden_cache
        self.seqs = []
        self.token_ids = []
        self.drafts = []
        self.num_drafts = []
        self.last_hidden = []    # per sequence: the head's output hidden state of its last row

    def begin(self, seqs, token_ids, num_drafts):
        self.seqs = seqs
        self.token_ids = [list(t) for t in token_ids]
        self.drafts = [[] for _ in seqs]
        self.num_drafts = list(num_drafts)
        self.last_hidden = [None] * len(seqs)

    def read_hidden_states(self, slots: torch.Tensor) -> torch.Tensor:    # as Eagle3Head.read_hidden_states
        return self.hidden_cache[torch.where(slots >= 0, slots, self.hidden_cache.size(0) - 1).long()]

    def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        bs = self.model.block_size
        c = get_context()
        if c.is_prefill:
            raise Violation("the head was called for prefill through propose")
        if c.slot_mapping.dtype != torch.int32 or c.context_lens.dtype != torch.int32:
            raise Violation(f"head step slot_mapping {c.slot_mapping.dtype} context_lens {c.context_lens.dtype}, not int32")
        ids, pos, slots = input_ids.tolist(), positions.tolist(), c.slot_mapping.tolist()
        hidden = hidden_states.view(-1).tolist()
        context_lens, block_tables = c.context_lens.tolist(), c.block_tables.tolist()
        q = len(ids) // len(self.seqs)
        if len(ids) != q * len(self.seqs) or len(pos) != len(ids) or len(slots) != len(ids) or len(hidden) != len(ids):
            raise Violation(f"head step inputs: {len(ids)} ids, {len(pos)} positions, {len(slots)} slots, {len(hidden)} hidden states for {len(self.seqs)} seqs")
        outputs = []
        step = max(map(len, self.drafts))
        for i, seq in enumerate(self.seqs):
            token_ids, drafts = self.token_ids[i], self.drafts[i]
            n = len(token_ids)    # the sequence's tokens and its drafts so far: the head's rows end at position n - 2
            rows = slice(i * q, (i + 1) * q)
            step_ids, step_pos, step_slots, step_hidden = ids[rows], pos[rows], slots[rows], hidden[rows]
            if len(drafts) >= self.num_drafts[i]:
                # A dummy row (see FakeDraftModel): its KV masked, its context its own, its proposal dropped.
                if any(s != -1 for s in step_slots):
                    raise Violation(f"seq {seq.seq_id}: head step {step} writes slots {step_slots} after its {self.num_drafts[i]} drafts")
                if context_lens[i] != n - 1 or any(p >= n for p in step_pos):
                    raise Violation(f"seq {seq.seq_id}: dummy head step {step} positions {step_pos} context_len {context_lens[i]} for {n} tokens")
                outputs.extend([[-1, 0]] * (q - 1) + [[0, 0]])
                continue
            truth = prefix_hashes(token_ids)
            head_truth = prefix_hashes(token_ids, DRAFT_SEED)
            if drafts:    # one row per sequence: the newest draft, at the position before the target scores it
                expected_ids, expected_pos = [drafts[-1]], [n - 2]
                if step_hidden[0] != self.last_hidden[i]:
                    raise Violation(f"seq {seq.seq_id}: draft step {len(drafts)} hidden state {step_hidden[0]} is not the head's own output {self.last_hidden[i]}")
            else:
                # The catch-up rows are at the last q positions before the newest token, each with
                # the token after it and the target's hidden state at it; the padded and the cached
                # rows have a masked slot, and anything passes there.
                positions_q = list(range(n - 1 - q, n - 1))
                expected_pos = [max(0, p) for p in positions_q]
                expected_ids = [token_ids[p + 1] if s != -1 else step_ids[j] for j, (p, s) in enumerate(zip(positions_q, step_slots))]
                if step_slots[-1] == -1:
                    raise Violation(f"seq {seq.seq_id}: the catch-up step does not write the head KV of the row before the newest token")
                for j, (p, s) in enumerate(zip(positions_q, step_slots)):
                    if s != -1 and step_hidden[j] != fits_int64(truth[p]):
                        raise Violation(f"seq {seq.seq_id}: catch-up row at position {p} read hidden state {step_hidden[j]}, not the target's {fits_int64(truth[p])}")
            if step_ids != expected_ids:
                raise Violation(f"seq {seq.seq_id}: head step {len(drafts)} inputs {step_ids} != {expected_ids}")
            if step_pos != expected_pos or context_lens[i] != n - 1:
                raise Violation(f"seq {seq.seq_id}: head step {len(drafts)} positions {step_pos} context_len {context_lens[i]} for {n} tokens")
            for j in range(q):
                p = n - 1 - q + j
                if step_slots[j] == -1:
                    continue
                if p < 0:
                    raise Violation(f"seq {seq.seq_id}: a padding row of the catch-up step writes slot {step_slots[j]}")
                self.model._write(self.model.draft_kv, seq, p, step_slots[j], (head_truth[p], token_ids[p + 1]))
            for p in range(n - 1):
                if not self.model.head_kv_matches(p, self.model.draft_kv[block_tables[i][p // bs] * bs + p % bs], (head_truth[p], token_ids[p + 1])):
                    raise Violation(f"seq {seq.seq_id}: head step {len(drafts)} read wrong head KV at position {p} (context_len {n - 1})")
            draft = draft_next_token(token_ids, truth[-1], head_truth[-1], self.model.draft_miss_mod(token_ids, len(drafts)))
            drafts.append(draft)
            token_ids.append(draft)
            self.last_hidden[i] = fits_int64(head_truth[-1])
            outputs.extend([[-1, 0]] * (q - 1) + [[draft, self.last_hidden[i]]])
        return torch.tensor(outputs, dtype=torch.int64)    # a row: the token (-1 unless the sequence's last row), then the hidden state fed back

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return FakeDraftModel.compute_logits(self, hidden_states)


class FakeModel:
    """Stands in for ModelRunner.call("run" / "run_async"). Like the GPU, it keeps the last step's
    sampled tokens, which the real prepare_decode reads for tokens still in flight."""

    def __init__(self, scheduler: Scheduler, block_size: int, num_blocks: int, check_tp: bool = True, num_speculative_tokens: int = 0, draft_model: bool = False,
                 eagle: bool = False):
        self.block_size = block_size
        self.kv = [None] * (num_blocks * block_size)
        self.draft_kv = [None] * (num_blocks * block_size) if draft_model or eagle else None    # the draft model's, or the head's
        self.eagle = eagle
        # The hidden cache an EAGLE-3 head reads (one "hidden" value per slot, plus the spare row
        # masked slots write to): the target writes its prefix hash at every row it scores.
        self.hidden_cache = torch.zeros(num_blocks * block_size + 1, 1, dtype=torch.int64) if eagle else None
        self.runner = make_runner(block_size, num_speculative_tokens=num_speculative_tokens)
        self.scheduler = scheduler
        if eagle:
            self.runner.eagle = True
            self.runner.draft_model = FakeEagleHead(self)
        elif draft_model:
            self.runner.draft_model = FakeDraftModel(self)
        self.block_manager = scheduler.block_manager
        self.check_tp = check_tp
        self.num_speculative_tokens = num_speculative_tokens
        self.mixed_draft_quality = False    # see draft_miss_mod
        self.steps_without_probe = 0        # consecutive decode steps with fewer than K drafts (adaptive k)
        self.num_capped_steps = 0           # the scheduler's count of steps with drafts off by batch size, as of the last decode step
        self.rejection_sampler = RejectionSampler()
        self.last_outputs = {}
        self.num_samples = collections.Counter()    # tokens sampled per seq, kept or not
        self.num_accepted = collections.Counter()   # decode steps by number of accepted drafts
        self.prefix_hashes = {}                     # seq_id -> prefix hashes of its tokens so far
        self.draft_prefix_hashes = {}
        self.checked_blocks = set()                 # (block_id, hash) pairs whose KV was checked

    def draft_miss_mod(self, token_ids, position):
        """How often the fake draft model misses at draft position 0..K-1: one draft in four, or with
        mixed_draft_quality every draft for prompts starting with an odd token and for the others one
        in eight at the first position, one in four at the second and one in two after that, so the
        batch's acceptance rates fall with the position, swing with its mix, and adaptive k moves
        both ways and stops at every position."""
        if not self.mixed_draft_quality:
            return 4
        if token_ids[0] % 2:
            return 1
        return (8, 4, 2)[min(position, 2)]

    def call(self, method_name, seqs, is_prefill):
        token_ids = self.run(seqs, is_prefill)
        for seq, t in zip(seqs, token_ids):
            if not is_prefill or seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.num_samples[seq.seq_id] += len(t) if isinstance(t, list) else 1
        self.last_outputs = {seq.seq_id: t for seq, t in zip(seqs, token_ids)}
        k = self.num_speculative_tokens
        if not k:
            self.runner.last_token_ids = torch.tensor(token_ids, dtype=torch.int64, device=DEVICE)
        else:    # a row per sequence, -1 padded, as ModelRunner.run_async keeps it
            rows = [t if isinstance(t, list) else [t] for t in token_ids]
            self.runner.last_token_ids = torch.tensor([row + [-1] * (k + 1 - len(row)) for row in rows], dtype=torch.int64, device=DEVICE)
        return token_ids if method_name == "run" else FakeOutput(token_ids)

    def run(self, seqs, is_prefill):
        meta = prepare(self.runner, seqs, is_prefill)
        if self.check_tp:
            # Ranks > 0 receive the sequences through pickle (write_shm/read_shm) and must build
            # exactly the same inputs as rank 0.
            remote_meta = prepare(self.runner, pickle.loads(pickle.dumps(seqs)), is_prefill)
            if remote_meta != meta:
                raise Violation(f"rank > 0 inputs differ from rank 0: {meta} vs {remote_meta}")
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        if len(ids) != len(pos) or len(slots) not in (len(ids), len(ids) - 1):    # a ragged draft batch has a scratch row the slots leave out
            raise Violation(f"length mismatch: ids={len(ids)} positions={len(pos)} slots={len(slots)}")
        return self._prefill(seqs, meta) if is_prefill else self._decode(seqs, meta)

    def _prefill(self, seqs, meta):
        bs = self.block_size
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        cu_q, cu_k, block_tables = meta["cu_seqlens_q"], meta["cu_seqlens_k"], meta["block_tables"]
        outputs = []
        for i, seq in enumerate(seqs):
            q0, q1 = cu_q[i], cu_q[i + 1]
            k_len = cu_k[i + 1] - cu_k[i]
            start = seq.num_cached_tokens
            if seq.num_inflight:
                raise Violation(f"seq {seq.seq_id}: prefill scheduled with a token in flight")
            if pos[q0:q1] != list(range(start, start + seq.num_scheduled_tokens)):
                raise Violation(f"seq {seq.seq_id}: positions {pos[q0:q1]}, expected to start at {start}")
            if k_len != start + seq.num_scheduled_tokens:
                raise Violation(f"seq {seq.seq_id}: seqlen_k {k_len} != {start} + {seq.num_scheduled_tokens}")
            truth = prefix_hashes(seq.token_ids[:k_len])
            # The draft model's prefill uses the same inputs, slots and block tables. The head's row
            # at position p takes token p + 1 (its KV: the draft hash of tokens[:p+2]), so the last
            # row of a complete prompt has nothing valid (None), and the head reads the target's
            # hidden states from the hidden cache, written here with the target's KV.
            if self.eagle:
                head_hashes = prefix_hashes(seq.token_ids[:k_len], DRAFT_SEED)
                head_values = [(head_hashes[p], seq.token_ids[p + 1]) for p in range(min(k_len, seq.num_tokens - 1))]
                caches = [(self.kv, truth, ""), (self.draft_kv, head_values, "head ")]
                for j in range(q0, q1):
                    if slots[j] != -1:
                        self.hidden_cache[slots[j], 0] = fits_int64(truth[pos[j]])
            else:
                caches = [(self.kv, truth, "")] + ([(self.draft_kv, prefix_hashes(seq.token_ids[:k_len], DRAFT_SEED), "draft ")] if self.draft_kv else [])
            for kv, values, name in caches:
                local = {}
                for j in range(q0, q1):
                    p = pos[j]
                    if ids[j] != seq.token_ids[p]:
                        raise Violation(f"seq {seq.seq_id}: wrong input id at position {p}")
                    self._write(kv, seq, p, slots[j], values[p] if p < len(values) else None)
                    if p < len(values):
                        local[p] = values[p]
                for p in range(len(values)):    # every position with valid KV (the head's end one earlier, see above)
                    if block_tables is not None:
                        value = kv[block_tables[i][p // bs] * bs + p % bs]
                    elif p in local:
                        value = local[p]
                    else:
                        raise Violation(f"seq {seq.seq_id}: needs cached position {p} but got no block tables")
                    if not (self.head_kv_matches(p, value, values[p]) if name == "head " else value == values[p]):
                        raise Violation(f"seq {seq.seq_id}: prefill read wrong {name}KV at position {p}")
            # compute_logits takes the hidden state at cu_seqlens_q[1:] - 1.
            outputs.append(next_token(seq.token_ids[:k_len], truth[pos[q1 - 1]]))
        return outputs

    def _propose(self, seqs, meta, token_ids, num_drafts, starts):
        """Run the real ModelRunner.propose with the fake draft model on the prepared inputs, and
        return the input ids with each sequence's draft rows (after its input row at starts[i])
        filled in."""
        draft_model = self.runner.draft_model
        draft_model.begin(seqs, token_ids, num_drafts)
        set_context(False, slot_mapping=torch.tensor(meta["slot_mapping"], dtype=torch.int32),
                    context_lens=torch.tensor(meta["context_lens"], dtype=torch.int32),
                    block_tables=torch.tensor(meta["block_tables"], dtype=torch.int32))
        self.runner.draft_inputs = tuple(torch.tensor(t, dtype=dtype) for t, dtype in zip(meta["draft_inputs"], meta["draft_dtypes"]))
        self.runner.draft_layout = tuple(torch.tensor(t) if t is not None else None for t in meta["draft_layout"])
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32)
        temperatures = self.runner.draft_temperatures(seqs, temperatures)    # None for a greedy batch, as run/run_async pass it
        ids = self.runner.propose(torch.tensor(meta["input_ids"], dtype=torch.int64), torch.tensor(meta["positions"], dtype=torch.int64), temperatures).tolist()
        draft_logits = self.runner.draft_logits
        if self.runner.sampled_drafts and temperatures is not None:
            # The logits the drafts were sampled from, one row per sequence and draft position,
            # each (the fake draft is certain of its token) peaking at the draft.
            if draft_logits is None or tuple(draft_logits.shape) != (len(seqs), max(num_drafts), VOCAB):
                raise Violation(f"draft_logits {None if draft_logits is None else tuple(draft_logits.shape)} for {len(seqs)} seqs and {max(num_drafts)} drafts")
            log_norms = self.runner.draft_log_norms
            if log_norms is None or tuple(log_norms.shape) != (len(seqs), max(num_drafts)) or not torch.allclose(log_norms, torch.logsumexp(draft_logits.float() / torch.tensor([max(seq.temperature, 1e-10) for seq in seqs]).view(-1, 1, 1), dim=-1)):
                raise Violation("draft_log_norms missing or not the log-sum-exp of draft_logits at the temperatures")
            peaks = draft_logits.argmax(dim=-1).tolist()
            for i, seq in enumerate(seqs):
                if peaks[i][:num_drafts[i]] != draft_model.drafts[i]:
                    raise Violation(f"seq {seq.seq_id}: draft_logits peak at {peaks[i][:num_drafts[i]]}, drafts {draft_model.drafts[i]}")
        elif draft_logits is not None or self.runner.draft_log_norms is not None:
            raise Violation("draft_logits set without sampled drafts" if not self.runner.sampled_drafts else "draft_logits set for a greedy batch")
        c = get_context()
        if c.is_prefill or c.context_lens.tolist() != meta["context_lens"] or c.slot_mapping.tolist() != meta["slot_mapping"]:
            raise Violation("propose did not restore the target's context")
        reset_context()
        for i, seq in enumerate(seqs):
            if len(draft_model.drafts[i]) != num_drafts[i]:
                raise Violation(f"seq {seq.seq_id}: the draft model ran {len(draft_model.drafts[i])} steps, not {num_drafts[i]}")
            if ids[starts[i] + 1:starts[i] + 1 + num_drafts[i]] != draft_model.drafts[i]:
                raise Violation(f"seq {seq.seq_id}: draft inputs {ids[starts[i] + 1:starts[i] + 1 + num_drafts[i]]} != proposed {draft_model.drafts[i]}")
        return ids, draft_model.drafts

    def _decode(self, seqs, meta):
        bs = self.block_size
        ids, pos, slots = meta["input_ids"], meta["positions"], meta["slot_mapping"]
        context_lens, block_tables = meta["context_lens"], meta["block_tables"]
        scheduler = self.scheduler
        K = self.num_speculative_tokens
        per_sequence = scheduler.speculative_per_sequence
        qs = [seq.num_scheduled_tokens for seq in seqs]    # 1 + each sequence's drafts in this step: none when the batch is large, any number with adaptive k
        q = max(qs)
        allowed = range(K + 1) if scheduler.speculative_min_acceptance is not None else (0, K)
        if (not per_sequence and any(seq.num_scheduled_tokens != q for seq in seqs)) or any(seq_q - 1 not in allowed for seq_q in qs):
            raise Violation(f"decode step with {qs} tokens per sequence")
        if scheduler.speculative_disable_by_batch_size is None and scheduler.speculative_min_acceptance is None and q != 1 + K:
            raise Violation(f"decode step with {q} tokens per sequence, expected {1 + K}")
        ragged = len(set(qs)) > 1
        pad = ragged and self.runner.ragged_mode == "pad"    # every sequence gets q rows, its own right-aligned behind dummy rows
        starts = [i * q + q - qs[i] if pad else sum(qs[:i]) for i in range(len(qs))]    # each sequence's first own row in the inputs
        capped, self.num_capped_steps = scheduler.num_capped_steps > self.num_capped_steps, scheduler.num_capped_steps
        if per_sequence and capped:
            # The batch-size rule still switches every sequence's drafts off. The scheduler counts
            # it so, as the number of running sequences it saw may have shrunk by preemption since.
            if q != 1 or len(scheduler.running) + len(scheduler.waiting) <= scheduler.speculative_disable_by_batch_size:
                raise Violation(f"decode step with {qs} tokens per sequence counted as capped with {len(scheduler.running)} running, {len(scheduler.waiting)} waiting")
            self.steps_without_probe += 1    # a capped step counts towards the batch's probe
        elif per_sequence:
            # Each sequence proposes the leading positions its own rates admit, and every position on
            # its first step; the probe is the batch's (counted here independently): after
            # PROBE_INTERVAL steps in which some sequence proposed fewer than K, every sequence
            # proposes K, in one uniform step.
            probe = self.steps_without_probe >= scheduler.PROBE_INTERVAL
            for seq, seq_q in zip(seqs, qs):
                expected = 0
                while expected < K and seq.acceptance[expected + 1] >= scheduler.speculative_min_acceptance:
                    expected += 1
                if probe:
                    expected = K
                if seq_q - 1 != expected:
                    raise Violation(f"seq {seq.seq_id}: {seq_q - 1} drafts, expected {expected} at acceptance {[round(a, 3) for a in seq.acceptance[1:]]} "
                                    f"after {self.steps_without_probe} steps without a probe")
            self.steps_without_probe = 0 if all(seq_q - 1 == K for seq_q in qs) else self.steps_without_probe + 1
        elif scheduler.speculative_min_acceptance is not None and scheduler.speculative_disable_by_batch_size is None:
            # Adaptive k alone decides: a step proposes the leading draft positions whose measured
            # acceptance rate reaches the threshold (computed here independently of the scheduler),
            # except a probe step with all K positions after PROBE_INTERVAL steps without one. The
            # scheduler's statistics are as they were when it scheduled this step: with async
            # scheduling the previous step is post-processed after this one is launched.
            expected = 0
            while expected < K and scheduler.acceptance[expected + 1] >= scheduler.speculative_min_acceptance:
                expected += 1
            if self.steps_without_probe >= scheduler.PROBE_INTERVAL:
                expected = K
            if q - 1 != expected:
                raise Violation(f"decode step with {q - 1} drafts, expected {expected} at acceptance {[round(a, 3) for a in scheduler.acceptance[1:]]} "
                                f"after {self.steps_without_probe} steps without a probe")
            self.steps_without_probe = 0 if q - 1 == K else self.steps_without_probe + 1
        n = len(seqs) * q if pad else sum(qs)
        if len(ids) != n + (ragged and not pad and self.draft_kv is not None) or len(slots) != n:    # a varlen ragged batch with a draft model has a scratch row
            raise Violation(f"{len(ids)} decode inputs and {len(slots)} slots for {len(seqs)} seqs with {qs} tokens each")
        # A ragged batch in the varlen mode runs the varlen attention: it carries the query and key
        # offsets, and the rejection sampler's layout; a uniform or padded one must not, so that it
        # takes the CUDA graphs (a padded one carries the sampler's layout only).
        if (not ragged or pad) and meta["cu_seqlens_q"] is not None or not ragged and meta["sampler_layout"] is not None:
            raise Violation("a uniform or padded decode batch carries varlen or ragged sampler inputs")
        if ragged:
            layout = ModelRunner.ragged_layout([seq_q - 1 for seq_q in qs], pad)
            if not pad:
                expected_k = [0]
                for c in context_lens:
                    expected_k.append(expected_k[-1] + c)
                if meta["cu_seqlens_q"] != starts + [n] or meta["cu_seqlens_k"] != expected_k or meta["max_seqlen_q"] != q or meta["max_seqlen_k"] < max(context_lens):
                    raise Violation(f"ragged batch offsets cu_seqlens_q={meta['cu_seqlens_q']} cu_seqlens_k={meta['cu_seqlens_k']} max_seqlen_q={meta['max_seqlen_q']} max_seqlen_k={meta['max_seqlen_k']} for {qs} tokens and context {context_lens}")
            if meta["sampler_layout"] != [layout[0].tolist(), layout[2].tolist()]:
                raise Violation(f"ragged sampler layout {meta['sampler_layout']} for {qs} tokens per sequence")
            if self.draft_kv is not None and meta["draft_layout"] != [t.tolist() for t in layout]:
                raise Violation(f"ragged draft layout {meta['draft_layout']} for {qs} tokens per sequence")
            if pad:
                # The dummy rows write nothing and sit at a valid position.
                for i, seq in enumerate(seqs):
                    for r in range(i * q, starts[i]):
                        if slots[r] != -1 or pos[r] < 0:
                            raise Violation(f"seq {seq.seq_id}: dummy row {r} has slot {slots[r]} at position {pos[r]}")
        # Each sequence's tokens, including those of the step still in flight (async scheduling).
        all_token_ids = []
        for seq in seqs:
            token_ids = list(seq.token_ids)
            if seq.num_inflight:
                inflight = self.last_outputs.get(seq.seq_id)
                if inflight is None or not 1 <= seq.num_inflight <= 1 + self.num_speculative_tokens:
                    raise Violation(f"seq {seq.seq_id}: {seq.num_inflight} tokens in flight but no step produced them")
                token_ids += inflight if isinstance(inflight, list) else [inflight]
            all_token_ids.append(token_ids)
        if self.draft_kv and q == 1:    # a step without drafts leaves the draft model alone
            if meta["draft_inputs"] is not None:
                raise Violation("draft inputs prepared for a step without drafts")
            all_drafts = [[] for _ in seqs]
        elif self.draft_kv:
            # In a ragged batch the sequences without drafts are left out of the catch-up (every
            # slot masked), so their draft cache lags like after a step without drafts.
            # The catch-up inputs: the last m tokens of every sequence, where m is 2 or the most tokens
            # a draft cache lacks, padded in front for a shorter sequence, each at its position and
            # slot, with the slot masked exactly for the cached and padding tokens. What a draft cache
            # lacks: the tokens after seq.num_draft_cached_tokens, or after an in-flight step with
            # drafts (which caught up itself) only its newest token, and the last draft too when all
            # of them were accepted. Whether every lacking token is written shows when the fake draft
            # model reads its cache (FakeDraftModel.__call__); here the slots are checked against that.
            # An EAGLE-3 head's rows sit one position behind: its catch-up rows are the positions
            # c .. n-2 (c: the head's valid positions, after an in-flight step with drafts that
            # step's starting n-1), each taking the token after it.
            shift = int(self.eagle)
            catchup_ids, catchup_pos, catchup_slots = meta["draft_inputs"]
            m = len(catchup_ids[0])
            cached = []
            for i, seq in enumerate(seqs):
                n_i = len(all_token_ids[i])
                if seq.num_inflight > 1 and self.eagle:
                    cached.append(n_i - len(self.last_outputs[seq.seq_id]) - 1)
                elif seq.num_inflight > 1:
                    cached.append(n_i - 1 - (len(self.last_outputs[seq.seq_id]) == seq.num_inflight))
                else:
                    cached.append(seq.num_draft_cached_tokens)
            # The CPU bounds what an in-flight step with drafts leaves the head lacking by all the tokens it could add.
            expected_m = max([2 - shift] + [seqs[i].num_inflight if self.eagle and seqs[i].num_inflight > 1 else len(all_token_ids[i]) - c - shift
                                            for i, c in enumerate(cached) if qs[i] > 1])
            if m != expected_m or any(len(row) != m for row in catchup_ids + catchup_pos + catchup_slots):
                raise Violation(f"catch-up inputs for {m} tokens per sequence, expected {expected_m}: {catchup_ids} {catchup_pos} {catchup_slots}")
            for i, seq in enumerate(seqs):
                n_i = len(all_token_ids[i])
                if catchup_pos[i] != [max(0, p) for p in range(n_i - shift - m, n_i - shift)]:
                    raise Violation(f"seq {seq.seq_id}: catch-up positions {catchup_pos[i]} for {n_i} tokens")
                for j, s in enumerate(catchup_slots[i]):
                    p = n_i - shift - m + j
                    if (s == -1) != (p < cached[i] or qs[i] == 1):
                        raise Violation(f"seq {seq.seq_id}: catch-up slot {s} at position {p} with {cached[i]} tokens in the draft cache and {qs[i] - 1} drafts")
                    if s == -1:
                        continue
                    if s != block_tables[i][p // bs] * bs + p % bs or catchup_ids[i][j] != all_token_ids[i][p + shift]:
                        raise Violation(f"seq {seq.seq_id}: catch-up token {catchup_ids[i][j]} at position {p} with slot {s}")
            ids, all_drafts = self._propose(seqs, meta, all_token_ids, [seq_q - 1 for seq_q in qs], starts)
        else:
            all_drafts = [seq.draft_token_ids for seq in seqs]
        truths, targets = [], []
        for i, seq in enumerate(seqs):
            rows = range(starts[i], starts[i] + qs[i])
            token_ids = all_token_ids[i]
            if ids[rows[0]] != token_ids[-1]:
                raise Violation(f"seq {seq.seq_id}: decode input {ids[rows[0]]} is not the newest token {token_ids[-1]}")
            drafts = ids[rows.start + 1:rows.stop]
            if drafts != all_drafts[i]:
                raise Violation(f"seq {seq.seq_id}: draft inputs {drafts} != proposed {all_drafts[i]}")
            n_i = len(token_ids)
            token_ids = token_ids + drafts
            if [pos[r] for r in rows] != list(range(n_i - 1, n_i + qs[i] - 1)) or context_lens[i] != n_i + qs[i] - 1:
                raise Violation(f"seq {seq.seq_id}: wrong decode inputs positions={[pos[r] for r in rows]} len={n_i} context_len={context_lens[i]}")
            truths.append(prefix_hashes(token_ids))
            for r in rows:
                self._write(self.kv, seq, pos[r], slots[r], truths[i][pos[r]])
                if self.eagle:    # the target's hidden state of every row it scores, drafts included
                    self.hidden_cache[slots[r], 0] = fits_int64(truths[i][pos[r]])
            targets.append([next_token(token_ids[:p + 1], truths[i][p]) for p in range(n_i - 1, n_i + qs[i] - 1)])
        for i, seq in enumerate(seqs):
            for p in range(context_lens[i]):
                if self.kv[block_tables[i][p // bs] * bs + p % bs] != truths[i][p]:
                    raise Violation(f"seq {seq.seq_id}: decode read wrong KV at position {p} (context_len {context_lens[i]})")
        if q == 1:
            return [t[0] for t in targets]
        # A target model that is certain of each next token, verified by the real rejection sampler
        # (compiled, on the GPU if there is one) with the draft tokens taken from the inputs, as
        # ModelRunner.run does.
        logits = torch.full((n, VOCAB), -1000.0, device=DEVICE)
        row_targets = [0] * n    # a dummy row (pad) is certain of token 0, which must not matter
        for i in range(len(seqs)):
            row_targets[starts[i]:starts[i] + qs[i]] = targets[i]
        logits[torch.arange(n), torch.tensor(row_targets)] = 0.0
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, device=DEVICE)
        ids = torch.tensor(ids, dtype=torch.int64, device=DEVICE)
        draft_logits = self.runner.draft_logits.to(DEVICE) if self.runner.draft_logits is not None else None    # one-hot-like: the sampler's q path must agree with the one-hot path
        draft_log_norms = self.runner.draft_log_norms.to(DEVICE) if draft_logits is not None else None
        if ragged:
            token_index, _, num_drafts = ModelRunner.ragged_layout([seq_q - 1 for seq_q in qs], pad)
            token_index, num_drafts = token_index.to(DEVICE), num_drafts.to(DEVICE)
            token_ids = self.rejection_sampler(logits, temperatures, ids[token_index[:, 1:]], token_index, num_drafts, draft_logits, draft_log_norms)
        else:
            token_ids = self.rejection_sampler(logits, temperatures, ids.view(len(seqs), q)[:, 1:], draft_logits=draft_logits, draft_log_norms=draft_log_norms)
        token_ids = accepted_token_ids(token_ids.tolist())
        for i, seq in enumerate(seqs):
            accepted = next(j for j in range(qs[i]) if j == qs[i] - 1 or all_drafts[i][j] != targets[i][j])
            if token_ids[i] != all_drafts[i][:accepted] + [targets[i][accepted]]:
                raise Violation(f"seq {seq.seq_id}: rejection sampler returned {token_ids[i]} for drafts {all_drafts[i]} and targets {targets[i]}")
            self.num_accepted[accepted] += 1
        return token_ids

    def check_hashed_blocks(self, seqs):
        """Every block registered for prefix caching must hold the KV of its tokens (with speculative
        decoding, rejected drafts leave stale KV that must not be hashed), in the draft model's
        cache too."""
        bs = self.block_size
        for seq in seqs:
            hashes = self.prefix_hashes.setdefault(seq.seq_id, [])
            if len(hashes) < seq.num_tokens:
                hashes += prefix_hashes(seq.token_ids[len(hashes):], hashes[-1] if hashes else TARGET_SEED)
            caches = [(self.kv, hashes, "")]
            if self.draft_kv:
                draft_hashes = self.draft_prefix_hashes.setdefault(seq.seq_id, [])
                if len(draft_hashes) < seq.num_tokens:
                    draft_hashes += prefix_hashes(seq.token_ids[len(draft_hashes):], draft_hashes[-1] if draft_hashes else DRAFT_SEED)
                if self.eagle:
                    caches.append((self.draft_kv, [(draft_hashes[p], seq.token_ids[p + 1]) for p in range(seq.num_tokens - 1)], "head "))
                else:
                    caches.append((self.draft_kv, draft_hashes, "draft "))
            for i, block_id in enumerate(seq.block_table):
                block = self.block_manager.blocks[block_id]
                if block.hash == -1 or (block_id, block.hash) in self.checked_blocks:
                    continue
                for kv, values, name in caches:
                    for p in range(i * bs, (i + 1) * bs):
                        value = kv[block_id * bs + p % bs]
                        if not (self.head_kv_matches(p, value, values[p]) if name == "head " else value == values[p]):
                            raise Violation(f"seq {seq.seq_id}: hashed block {block_id} holds wrong {name}KV at position {p}")
                self.checked_blocks.add((block_id, block.hash))

    def head_kv_matches(self, p, value, expected):
        """Whether the head's KV read at position p is right: (the draft hash of tokens[:p+1], token
        p + 1), except at the last position of a block, where the row of a block shared through the
        prefix cache was computed with the next token of the sequence that hashed the block."""
        if value == expected:
            return True
        bs = self.block_size
        return p % bs == bs - 1 and value is not None and value[0] == expected[0]

    def _write(self, kv, seq, p, slot, value):
        bs = self.block_size
        block_id, offset = divmod(slot, bs)
        if p // bs >= len(seq.block_table) or seq.block_table[p // bs] != block_id or offset != p % bs:
            raise Violation(f"seq {seq.seq_id}: position {p} written to slot {slot} outside its blocks {seq.block_table}")
        block = self.block_manager.blocks[block_id]
        if block.hash != -1:
            raise Violation(f"seq {seq.seq_id}: position {p} written into hashed block {block_id}")
        if block.ref_count != 1:
            raise Violation(f"seq {seq.seq_id}: position {p} written into shared block {block_id} (ref_count {block.ref_count})")
        kv[slot] = value


def check_block_manager(block_manager, live_seqs):
    n = len(block_manager.blocks)
    refs = [0] * n
    for seq in live_seqs:
        for block_id in seq.block_table:
            refs[block_id] += 1
    for block_id in range(n):
        if block_manager.blocks[block_id].ref_count != refs[block_id]:
            raise Violation(f"block {block_id}: ref_count {block_manager.blocks[block_id].ref_count} != {refs[block_id]} live references")
    free = list(block_manager.free_block_ids)
    used = block_manager.used_block_ids
    if len(set(free)) != len(free):
        raise Violation("duplicate block in the free list")
    if set(free) & used or set(free) | used != set(range(n)):
        raise Violation("free and used blocks do not partition the pool")
    if any(refs[block_id] for block_id in free):
        raise Violation("a free block is still referenced")
    for h, block_id in block_manager.hash_to_block_id.items():
        if block_manager.blocks[block_id].hash != h:
            raise Violation(f"hash_to_block_id maps {h} to block {block_id} with hash {block_manager.blocks[block_id].hash}")


def run_engine(engine, requests, max_steps=50000):
    """Feed (arrival_step, prompt, sampling_params) requests to the engine, check the block manager
    after every step, and return {request index: completion token ids}."""
    scheduler = engine.scheduler
    pending = sorted(range(len(requests)), key=lambda i: requests[i][0])
    seqs = {}
    outputs = {}
    step = 0
    while True:
        while pending and requests[pending[0]][0] <= step:
            i = pending.pop(0)
            seq = Sequence(requests[i][1], requests[i][2])
            seqs[seq.seq_id] = (i, seq)
            scheduler.add(seq)
        if engine.is_finished():
            if not pending:
                break
            step += 1
            continue
        for seq_id, token_ids in engine.step()[0]:
            if seqs[seq_id][0] in outputs:
                raise Violation(f"seq {seq_id} finished twice")
            outputs[seqs[seq_id][0]] = token_ids
        live_seqs = [s for s in [*scheduler.running, *scheduler.waiting] if s.block_table]
        check_block_manager(scheduler.block_manager, live_seqs)
        engine.model_runner.check_hashed_blocks(live_seqs)
        # An in-flight prefill chunk's KV is counted in num_cached_tokens already but hashed only
        # once the step lands (async scheduling).
        inflight = {seq: num_tokens for seq, is_prefill, _, num_tokens, _ in (engine.pending[0] if engine.pending else []) if is_prefill}
        for seq in live_seqs:
            # Every full block of valid KV is registered for prefix caching, except the one the
            # draft model has not caught up on yet.
            num_valid = min(seq.num_cached_tokens, seq.num_draft_cached_tokens) - inflight.get(seq, 0)
            num_hashed = sum((i + 1) * seq.block_size <= num_valid for i in range(len(seq.block_table)))
            if any(scheduler.block_manager.blocks[block_id].hash == -1 for block_id in seq.block_table[:num_hashed]):
                raise Violation(f"seq {seq.seq_id}: a full block with valid KV is not hashed")
        k = engine.model_runner.num_speculative_tokens
        for _, seq in seqs.values():
            if seq.num_completion_tokens > seq.max_tokens:
                raise Violation(f"seq {seq.seq_id}: more than max_tokens completion tokens")
            # Speculative decoding may produce up to k tokens past EOS or max_tokens in the last step.
            # Async scheduling may run one step too many (1 + k tokens) for a seq that hit EOS, never
            # more, and none for a seq that reached max_tokens.
            if seq.is_finished:
                wasted = engine.model_runner.num_samples[seq.seq_id] - seq.num_completion_tokens
                allowed = k + (1 + k) * (engine.async_scheduling and seq.num_completion_tokens < seq.max_tokens)
                if wasted > allowed:
                    raise Violation(f"seq {seq.seq_id}: {wasted} sampled tokens thrown away")
        step += 1
        if step > max_steps:
            raise Violation("engine did not finish")
    block_manager = scheduler.block_manager
    if len(block_manager.free_block_ids) != len(block_manager.blocks) or block_manager.used_block_ids:
        raise Violation("blocks leaked after all requests finished")
    return outputs


def random_requests(rng, block_size, capacity, pressure):
    """Prompts that share prefixes of various lengths, including ones that share later blocks but
    not the first (prefix hashes must chain), with max_tokens capped so a lone request fits."""
    bs = block_size
    bases = [[rng.randrange(1, VOCAB) for _ in range(rng.randint(1, 4 * bs + 1))] for _ in range(3)]
    requests = []
    for _ in range(rng.randint(4, 16) if pressure else rng.randint(1, 12)):
        r = rng.random()
        if r < 0.15:
            base = rng.choice(bases)
            prompt = [rng.randrange(1, VOCAB) for _ in range(bs)] + base[bs:] + [rng.randrange(1, VOCAB) for _ in range(rng.randint(0, bs))]
        elif r < 0.5:
            base = rng.choice(bases)
            k = max(1, rng.choice([len(base), len(base) // bs * bs, rng.randint(1, len(base))]))
            prompt = base[:k] + [rng.randrange(1, VOCAB) for _ in range(rng.choice([0, 0, 1, bs - 1, bs, rng.randint(0, 2 * bs)]))]
        else:
            length = max(1, rng.choice([bs - 1, bs, bs + 1, 2 * bs, 2 * bs + 1, rng.randint(1, 5 * bs)]))
            lowest = 0 if rng.random() < 0.3 else 1    # EOS in the prompt can be copied into drafts
            prompt = [rng.randrange(lowest, VOCAB) for _ in range(length)]
        max_tokens = rng.choice([1, 2, bs, bs + 1, rng.randint(1, 3 * bs)] + ([rng.randint(bs, 6 * bs)] * 3 if pressure else []))
        if len(prompt) + max_tokens > capacity:
            prompt = prompt[:max(1, capacity - max_tokens)]
            max_tokens = min(max_tokens, capacity - len(prompt))
        if rng.random() < 0.15:    # EOS near the end of the prompt shows up in periodic outputs and their drafts
            prompt[-rng.randint(1, min(3, len(prompt)))] = EOS
        sp = SamplingParams(temperature=rng.choice([0.0, 1.0]), max_tokens=max_tokens, ignore_eos=rng.random() < 0.5)
        arrival = 0 if rng.random() < 0.6 else rng.randint(0, 30)
        requests.append((arrival, prompt, sp))
    return requests


def set_acceptance_switch(engine, rng):
    """A short window and probe interval, so k changes many times within a case."""
    engine.scheduler.ACCEPTANCE_WINDOW = rng.randint(2, 40)
    engine.scheduler.ACCEPTANCE_MIN_STEPS = rng.randint(1, 4)
    engine.scheduler.ACCEPTANCE_WINDOW_PER_SEQUENCE = rng.randint(1, 12)
    engine.scheduler.PROBE_INTERVAL = rng.randint(1, 6)
    engine.model_runner.mixed_draft_quality = True


def run_random_case(seed, async_scheduling=False, num_speculative_tokens=0, draft_model=False, adaptive=False, acceptance=False, per_sequence=False, eagle=False):
    rng = random.Random(seed)
    bs = rng.choice([2, 3, 4, 8])
    Sequence.block_size = bs
    pressure = rng.random() < 0.5
    num_blocks = rng.randint(2, 10) if pressure else rng.randint(3, 40)
    # A lone request must fit, and a decode step writes num_speculative_tokens slots past its last token.
    num_blocks = max(num_blocks, (num_speculative_tokens + 2 + bs - 1) // bs)
    max_num_seqs = rng.randint(1, 8)
    max_num_batched_tokens = max(1, rng.choice([1, 2, bs - 1, bs, bs + 1, 2 * bs, 3 * bs + 1, rng.randint(1, 10 * bs), 10 ** 6]))
    # Adaptive: drafts only while at most 1-3 sequences run, so steps with and without drafts alternate.
    # Acceptance: k follows the per-position acceptance rates (half the requests get good drafts).
    engine = make_engine(num_blocks, bs, max_num_seqs, max_num_batched_tokens, async_scheduling, num_speculative_tokens, draft_model,
                         speculative_disable_by_batch_size=rng.randint(1, 3) if adaptive else None,
                         speculative_min_acceptance=rng.choice([0.2, 0.5, 0.8]) if acceptance else None, speculative_per_sequence=per_sequence,
                         speculative_ragged=rng.choice(["varlen", "pad"]) if per_sequence else "varlen",
                         speculative_sampled_drafts=rng.random() < 0.7, eagle=eagle)
    if acceptance:
        set_acceptance_switch(engine, rng)
    requests = random_requests(rng, bs, num_blocks * bs - num_speculative_tokens, pressure)
    outputs = run_engine(engine, requests)
    for i, (_, prompt, sp) in enumerate(requests):
        expected = reference(prompt, sp)
        if outputs.get(i) != expected:
            raise Violation(f"request {i}: output {outputs.get(i)} != reference {expected}")


def run_random_case_block_256(seed, async_scheduling=False, num_speculative_tokens=0, draft_model=False, adaptive=False, acceptance=False, per_sequence=False, eagle=False):
    """Same checks with the default block size and realistic token budgets."""
    rng = random.Random(seed)
    bs = 256
    Sequence.block_size = bs
    num_blocks = rng.randint(4, 16)
    capacity = num_blocks * bs - num_speculative_tokens
    max_num_batched_tokens = rng.choice([256, 300, 511, 512, 1000, 16384])
    engine = make_engine(num_blocks, bs, rng.randint(1, 6), max_num_batched_tokens, async_scheduling, num_speculative_tokens, draft_model,
                         speculative_disable_by_batch_size=rng.randint(1, 3) if adaptive else None,
                         speculative_min_acceptance=rng.choice([0.2, 0.5, 0.8]) if acceptance else None, speculative_per_sequence=per_sequence,
                         speculative_ragged=rng.choice(["varlen", "pad"]) if per_sequence else "varlen",
                         speculative_sampled_drafts=rng.random() < 0.7, eagle=eagle)
    if acceptance:
        set_acceptance_switch(engine, rng)
    base = [rng.randrange(1, VOCAB) for _ in range(1200)]
    requests = []
    for _ in range(rng.randint(2, 6)):
        k = rng.choice([255, 256, 257, 511, 512, 513, 768, rng.randint(1, 1200)])
        prompt = (base[:k] if rng.random() < 0.6 else [rng.randrange(1, VOCAB) for _ in range(k)]) + [rng.randrange(1, VOCAB) for _ in range(rng.choice([0, 1, 5]))]
        max_tokens = rng.choice([1, 2, 300, rng.randint(1, 600)])
        if len(prompt) + max_tokens > capacity:
            max_tokens = max(1, capacity - len(prompt))
            prompt = prompt[:capacity - max_tokens]
        requests.append((0, prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=rng.random() < 0.7)))
    outputs = run_engine(engine, requests)
    for i, (_, prompt, sp) in enumerate(requests):
        if outputs.get(i) != reference(prompt, sp):
            raise Violation(f"request {i}: output differs from reference")
