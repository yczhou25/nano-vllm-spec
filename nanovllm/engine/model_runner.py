import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle3 import Eagle3Head
from nanovllm.layers.sampler import DraftSampler, RejectionSampler, Sampler, accepted_token_ids
from nanovllm.utils.context import set_context, get_context, restore_context, reset_context
from nanovllm.utils.loader import load_model


class AsyncOutput:
    """A step's sampled tokens, copied to the host in the background: one per sequence, or with
    speculative decoding a row per sequence (RejectionSampler's accepted tokens, -1 padded)."""

    def __init__(self, token_ids: torch.Tensor):
        self.token_ids = torch.empty(token_ids.shape, dtype=token_ids.dtype, pin_memory=True)
        self.token_ids.copy_(token_ids, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record()

    def result(self) -> list[int] | list[list[int]]:
        self.event.synchronize()
        token_ids = self.token_ids.tolist()
        return accepted_token_ids(token_ids) if token_ids and isinstance(token_ids[0], list) else token_ids


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.num_speculative_tokens = config.num_speculative_tokens
        self.rank = rank
        self.event = event
        self.last_token_ids = None    # last run_async step's sampled tokens, still on the GPU (see AsyncOutput)
        self.draft_model = None
        self.eagle = config.speculative_method == "eagle3"    # the draft model is an EAGLE-3 head (models/eagle3.py), drafting from the target's hidden states
        self.draft_inputs = None      # the draft model's decode inputs, prepared with the target's
        self.ragged_mode = config.speculative_ragged    # how a ragged batch runs: "varlen" or "pad" (see prepare_decode)
        self.sampled_drafts = config.speculative_sampled_drafts    # a draft model samples its drafts (see propose), with the distributions kept for the sampler
        self.draft_logits = None      # (batch_size, k, vocab_size): the draft model's logits each draft was sampled from, in a step that sampled them
        self.draft_log_norms = None   # (batch_size, k): their log-sum-exp at the temperature (DraftSampler), with which the sampler reads q
        self.draft_layout = None      # where each sequence's drafts go in the decode inputs (see ragged_layout)
        self.sampler_layout = None    # the same for RejectionSampler, in a ragged batch only
        self.graphs = None            # CUDA graphs by decode batch size, once captured (see capture_cudagraph)
        self.draft_graphs = None
        self.graph_pool = None

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        try:
            torch.cuda.set_device(rank)
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(hf_config.dtype)
            torch.set_default_device("cuda")
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)
            if config.speculative_model is not None:
                self.draft_model = Eagle3Head(config.draft_hf_config) if self.eagle else Qwen3ForCausalLM(config.draft_hf_config)
                load_model(self.draft_model, config.speculative_model)
                if self.eagle:    # the target hands the head its aux layers' inputs every forward
                    self.model.model.aux_hidden_state_layers = config.draft_hf_config.aux_hidden_state_layers
                    self.model.model.aux_hidden_state_hook = self.draft_model.store_hidden_states
            self.sampler = Sampler()
            self.rejection_sampler = RejectionSampler()
            self.draft_sampler = DraftSampler()
            torch.set_default_device("cpu")
            torch.set_default_dtype(default_dtype)
            # The warmup and the graph capture run under the same default device and dtype as the
            # engine does later: torch.compile guards on the default device, so a layer compiled
            # while it was "cuda" is compiled again at the first real request.
            self.warmup_model()
            self.warmup_samplers()
            self.allocate_kv_cache()
            if not self.enforce_eager:
                self.capture_cudagraph()
        except Exception:
            dist.destroy_process_group()    # or the next engine in this process could not start
            raise

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        del self.graphs, self.draft_graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        for event in self.event:
            while event.is_set():    # with async scheduling a worker may not have read the last call yet
                pass
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def decode_query_lengths(self) -> list[int]:
        """Tokens per sequence a decode step can have: 1 + the drafts, 1 for a step without drafts
        (a large batch, see Scheduler.schedule), and with adaptive k anything in between."""
        config = self.config
        q = 1 + self.num_speculative_tokens
        qs = {q}
        if config.speculative_disable_by_batch_size is not None:
            qs.add(1)
        if config.speculative_min_acceptance is not None:
            qs.update(range(1, q))
        return sorted(qs)

    def warmup_samplers(self):
        """Compile the samplers now rather than in the first decode steps. They are compiled for
        symbolic shapes, so a few rows per query length cover every batch size, with one exception:
        torch.compile specializes sizes 0 and 1, so a batch of one sequence (the tail of every
        batch) and a step with one draft are shapes of their own, compiled here as well. The inputs
        are like the real ones in everything else the compiled code guards on: dtype, device,
        strides (the drafts are a column slice of the step's input ids) and grad mode. A draft
        model that samples its drafts passes their logits (draft_logits), another branch, and
        its DraftSampler is compiled here too."""
        hf_config = self.config.hf_config
        sampled = self.draft_model is not None and self.sampled_drafts
        with torch.inference_mode():
            for batch_size in (8, 1):
                temperatures = torch.ones(batch_size, dtype=torch.float32, device="cuda")
                if sampled:
                    self.draft_sampler(torch.zeros(batch_size, self.config.draft_hf_config.vocab_size, dtype=hf_config.dtype, device="cuda"), temperatures)
                for q in self.decode_query_lengths():
                    logits = torch.zeros(batch_size * q, hf_config.vocab_size, dtype=hf_config.dtype, device="cuda")
                    if q == 1:
                        self.sampler(logits, temperatures)
                    else:
                        input_ids = torch.zeros(batch_size, q, dtype=torch.int64, device="cuda")
                        self.rejection_sampler(logits, temperatures, input_ids[:, 1:])
                        if sampled:
                            draft_logits = torch.zeros(batch_size, q - 1, hf_config.vocab_size, dtype=hf_config.dtype, device="cuda")
                            self.rejection_sampler(logits, temperatures, input_ids[:, 1:], draft_logits=draft_logits, draft_log_norms=torch.zeros(batch_size, q - 1, device="cuda"))
            if self.config.speculative_per_sequence:
                # Ragged batches (at least two sequences, every number of drafts up to K) take the
                # sampler's other branch, with the drafts gathered by the batch's layout.
                batch_size = 8
                temperatures = torch.ones(batch_size, dtype=torch.float32, device="cuda")
                for k in range(1, self.num_speculative_tokens + 1):
                    num_drafts = [k] + [j % (k + 1) for j in range(batch_size - 1)]
                    token_index, _, num_drafts = self.ragged_layout(num_drafts)
                    n = sum(seq_drafts + 1 for seq_drafts in num_drafts.tolist())
                    logits = torch.zeros(n, hf_config.vocab_size, dtype=hf_config.dtype, device="cuda")
                    input_ids = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
                    token_index, num_drafts = token_index.cuda(), num_drafts.cuda()
                    self.rejection_sampler(logits, temperatures, input_ids[token_index[:, 1:]], token_index, num_drafts)
                    if sampled:
                        draft_logits = torch.zeros(batch_size, k, hf_config.vocab_size, dtype=hf_config.dtype, device="cuda")
                        self.rejection_sampler(logits, temperatures, input_ids[token_index[:, 1:]], token_index, num_drafts, draft_logits, torch.zeros(batch_size, k, device="cuda"))

    def kv_cache_block_bytes(self, hf_config) -> int:
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

    def bind_kv_cache(self, model: Qwen3ForCausalLM, hf_config, num_blocks: int) -> torch.Tensor:
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_cache = torch.empty(2, hf_config.num_hidden_layers, num_blocks, self.block_size, num_kv_heads, head_dim, dtype=hf_config.dtype, device="cuda")
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        return kv_cache

    def allocate_kv_cache(self):
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # The draft model keeps its own KV cache, with the same block ids as the target's.
        block_bytes = self.kv_cache_block_bytes(config.hf_config)
        if self.draft_model is not None:
            block_bytes += self.kv_cache_block_bytes(config.draft_hf_config)
            if self.eagle:    # and the hidden cache: the target's fused hidden state of every position (see Eagle3Head)
                block_bytes += self.block_size * config.draft_hf_config.hidden_size * config.draft_hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = self.bind_kv_cache(self.model, config.hf_config, config.num_kvcache_blocks)
        if self.draft_model is not None:
            self.draft_kv_cache = self.bind_kv_cache(self.draft_model, config.draft_hf_config, config.num_kvcache_blocks)
            if self.eagle:
                self.draft_model.hidden_cache = torch.zeros(config.num_kvcache_blocks * self.block_size + 1, config.draft_hf_config.hidden_size,
                                                            dtype=config.draft_hf_config.dtype, device="cuda")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @staticmethod
    def ragged_layout(num_drafts: list[int], pad: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Where each sequence's rows are in a decode step's inputs, which hold every sequence's
        input token and then its drafts, one row each. Without pad ("varlen"), sequence i has
        1 + num_drafts[i] rows and a scratch row follows the last sequence; with pad, every sequence
        has 1 + max(num_drafts) rows, its own right-aligned behind k - num_drafts[i] dummy rows.
        Returns token_index (batch_size, 1 + max drafts), the row of each sequence's input token
        and drafts, padded by repeating its last row; write_index (batch_size, max drafts), the rows
        its drafts are written to, with the scratch row (or one of its dummy rows) for the drafts it
        does not have; and num_drafts (batch_size,). A batch whose sequences all have the same
        number of drafts is a (batch_size, 1 + k) grid either way, and the samplers take it as such."""
        k = max(num_drafts)
        n = sum(num_drafts) + len(num_drafts)
        token_index, write_index, row = [], [], 0
        for num in num_drafts:
            front = k - num if pad else 0
            token_index.append([row + front + min(j, num) for j in range(k + 1)])
            write_index.append([row + front + j if j <= num else (row + j - num - 1 if pad else n) for j in range(1, k + 1)])
            row += 1 + (k if pad else num)
        return torch.tensor(token_index, dtype=torch.int64), torch.tensor(write_index, dtype=torch.int64), torch.tensor(num_drafts, dtype=torch.int64)

    def prepare_decode(self, seqs: list[Sequence]):
        self.draft_layout = self.sampler_layout = None
        if self.draft_model is not None:
            return self.prepare_decode_with_draft_model(seqs)
        input_ids = []
        inflight_rows = []
        positions = []
        slot_mapping = []
        context_lens = []
        cu_seqlens_q = [0]
        num_drafts = [len(seq.draft_token_ids) for seq in seqs]
        ragged = len(set(num_drafts)) > 1    # speculative_per_sequence: sequences with different numbers of drafts
        pad = ragged and self.ragged_mode == "pad"
        for seq in seqs:
            # With async scheduling the newest token may still be on the GPU, in the last step's output.
            # With speculative decoding the draft tokens follow it, and all of them are scored at once.
            seq_len = len(seq) + seq.num_inflight
            if pad:
                # Dummy rows in front, so that the sequence's own rows end the fixed-width window
                # (see prepare_decode_with_draft_model): nothing is written for them.
                front = max(num_drafts) - len(seq.draft_token_ids)
                input_ids.extend([0] * front)
                positions.extend([0] * front)
                slot_mapping.extend([-1] * front)
            input_ids.append(0 if seq.num_inflight else seq.last_token)
            input_ids.extend(seq.draft_token_ids)
            inflight_rows.append(seq.inflight_row if seq.num_inflight else -1)
            for position in range(seq_len - 1, seq_len + len(seq.draft_token_ids)):
                positions.append(position)
                slot_mapping.append(seq.block_table[position // self.block_size] * self.block_size + position % self.block_size)
            context_lens.append(positions[-1] + 1)
            cu_seqlens_q.append(len(input_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if any(row >= 0 for row in inflight_rows):
            inflight_rows = torch.tensor(inflight_rows, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            input_ids = torch.where(inflight_rows >= 0, self.last_token_ids[inflight_rows.clamp_min(0)], input_ids)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        if ragged and not pad:
            self.set_ragged_context(seqs, num_drafts, cu_seqlens_q, slot_mapping, context_lens, block_tables)
        else:
            if pad:
                token_index, _, num_drafts_tensor = self.ragged_layout(num_drafts, pad=True)
                self.sampler_layout = token_index.cuda(non_blocking=True), num_drafts_tensor.cuda(non_blocking=True)
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def set_ragged_context(self, seqs: list[Sequence], num_drafts: list[int], cu_seqlens_q: list[int], slot_mapping: torch.Tensor, context_lens: torch.Tensor, block_tables: torch.Tensor):
        """The context of a ragged decode batch (speculative_per_sequence) in the "varlen" mode:
        its sequences have different numbers of drafts, so the attention runs the varlen kernel
        over the block tables instead of the fixed-width decode kernel and the CUDA graphs, and
        RejectionSampler gets the batch's layout. (The "pad" mode keeps the fixed-width kernel, see
        prepare_decode_with_draft_model.) The key lengths are cumulated on the GPU, where with async scheduling the
        in-flight tokens are only known; their maximum is bounded from the CPU side."""
        token_index, _, num_drafts = self.ragged_layout(num_drafts)
        self.sampler_layout = token_index.cuda(non_blocking=True), num_drafts.cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=context_lens.device)
        cu_seqlens_k[1:] = context_lens.cumsum(0)
        max_seqlen_q = 1 + max(seq.num_scheduled_tokens - 1 for seq in seqs)
        max_seqlen_k = max(len(seq) + seq.num_inflight + seq.num_scheduled_tokens - 1 for seq in seqs)
        set_context(False, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

    def prepare_decode_with_draft_model(self, seqs: list[Sequence]):
        """Decode inputs for speculative decoding with a draft model, computed on the GPU: with
        async scheduling the number of tokens the in-flight step added, and so every position, is
        only known there. In a step with drafts also prepares the draft model's catch-up inputs
        (see propose): the tokens its KV cache lacks, many after steps without drafts."""
        num_drafts = [seq.num_scheduled_tokens - 1 for seq in seqs]    # per sequence: up to num_speculative_tokens, 0 for a large batch
        k = max(num_drafts)    # drafts in this step; with speculative_per_sequence some sequences have fewer (a ragged batch)
        block_size = self.block_size

        def to_cuda(values, dtype):
            return torch.tensor(values, dtype=dtype, pin_memory=True).cuda(non_blocking=True)

        num_tokens = to_cuda([len(seq) for seq in seqs], torch.int64)    # as known on the CPU
        last_token = to_cuda([seq.last_token for seq in seqs], torch.int64)
        block_tables = self.prepare_block_tables(seqs)
        inflight_rows = [seq.inflight_row if seq.num_inflight else -1 for seq in seqs]
        output = None
        if any(row >= 0 for row in inflight_rows):
            # The in-flight step's output: per sequence its accepted tokens, then -1.
            inflight_rows = to_cuda(inflight_rows, torch.int64)
            output = self.last_token_ids[inflight_rows.clamp_min(0)]
            num_inflight = torch.where(inflight_rows >= 0, (output >= 0).sum(dim=1), 0)
            num_tokens = num_tokens + num_inflight
            newest = output.gather(1, (num_inflight - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
            last_token = torch.where(num_inflight >= 1, newest, last_token)
        # The target scores the newest token and k drafts (propose fills them in) at positions
        # num_tokens - 1 .. num_tokens + k - 1: a row per sequence here, of which a sequence with
        # fewer drafts uses the first 1 + num_drafts (the others repeat its last position, as it
        # has no blocks past it, and are dropped below) in the "varlen" mode. In the "pad" mode its
        # rows are the last 1 + num_drafts instead, behind dummy rows (position clamped to 0, slot
        # masked, input 0): the fixed-width attention aligns every sequence's last row with its
        # context length, so its own rows see exactly their keys and the dummy rows see fewer, as
        # in the draft model's catch-up step. That keeps the kernel and the CUDA graphs; the dummy
        # rows are the price.
        pad = self.ragged_mode == "pad" and len(set(num_drafts)) > 1
        token_index, write_index, num_drafts_tensor = self.ragged_layout(num_drafts, pad)
        num_drafts_tensor = num_drafts_tensor.cuda(non_blocking=True)
        cols = torch.arange(k + 1, device=num_tokens.device)
        if pad:
            front = (k - num_drafts_tensor).unsqueeze(1)
            positions = ((num_tokens - 1).unsqueeze(1) + cols - front).clamp_min(0)
            slot_mapping = block_tables.gather(1, positions // block_size) * block_size + positions % block_size
            slot_mapping = torch.where(cols >= front, slot_mapping, -1).to(torch.int32)
            input_ids = torch.zeros_like(positions)
            input_ids[torch.arange(len(seqs), device=positions.device), front.squeeze(1)] = last_token
        else:
            positions = (num_tokens - 1).unsqueeze(1) + torch.minimum(cols, num_drafts_tensor.unsqueeze(1))
            slot_mapping = (block_tables.gather(1, positions // block_size) * block_size + positions % block_size).to(torch.int32)
            input_ids = torch.cat([last_token.unsqueeze(1), torch.zeros_like(positions[:, 1:])], dim=1)
        self.draft_inputs = None
        if k:
            # Before drafting, the draft model catches up on the tokens its KV cache lacks: the
            # newest one, the last draft of a fully accepted step, and everything sampled in steps
            # without drafts (see Scheduler.postprocess). It scores the last m tokens of every
            # sequence, m being the most any of them lacks (at least 2, which a CUDA graph covers);
            # tokens it has cached get a masked slot, and a sequence shorter than m is padded in
            # front. With async scheduling the in-flight step's tokens are appended on the GPU.
            # A sequence without drafts in this step is left out (every slot masked): its cache
            # falls behind as in a step without drafts, and the scheduler counts it so.
            cached = [seq.num_draft_cached_tokens for seq in seqs]
            if self.eagle:
                # The head's KV cache holds positions 0 .. c-1 and lacks the rows at c .. n-2, each
                # taking the token after its position: the last n-1-c tokens, read with the
                # target's hidden states at c .. n-2 from the hidden cache (see Eagle3Head). An
                # in-flight step with drafts leaves c at its own n-1, so its rows are the tokens it
                # accepted, at most all it could add.
                m = max([1] + [seq.num_inflight if seq.num_inflight > 1 else len(seq) + seq.num_inflight - c - 1 for seq, c, num in zip(seqs, cached, num_drafts) if num])
            else:
                m = max([2] + [2 if seq.num_inflight > 1 else len(seq) + seq.num_inflight - c for seq, c, num in zip(seqs, cached, num_drafts) if num])
            tail = to_cuda([([0] * m + seq.token_ids[c:])[-m:] for seq, c in zip(seqs, cached)], torch.int64)
            cached = to_cuda(cached, torch.int64)
            if output is not None:
                # An in-flight step with drafts caught up itself; afterwards its draft cache lacks
                # the newest token, and the last draft too when all of them were accepted.
                spec_inflight = to_cuda([seq.num_inflight > 1 for seq in seqs], torch.bool)    # in flight: the newest token and its drafts
                max_inflight = to_cuda([seq.num_inflight for seq in seqs], torch.int64)
                if self.eagle:    # the head's rows end one position before the newest token of the step before
                    cached = torch.where(spec_inflight, num_tokens - num_inflight - 1, cached)
                else:
                    cached = torch.where(spec_inflight, num_tokens - 1 - (num_inflight == max_inflight).long(), cached)
                tokens = torch.cat([tail, output.clamp_min(0)], dim=1)
                tail = tokens.gather(1, num_inflight.unsqueeze(1) + torch.arange(m, device=tokens.device))
            catchup_positions = (num_tokens - m - int(self.eagle)).unsqueeze(1) + torch.arange(m, device=num_tokens.device)
            padded_positions = catchup_positions.clamp_min(0)
            catchup_slots = block_tables.gather(1, padded_positions // block_size) * block_size + padded_positions % block_size
            drafted = to_cuda([num > 0 for num in num_drafts], torch.bool)
            catchup_slots = torch.where((catchup_positions >= cached.unsqueeze(1)) & drafted.unsqueeze(1), catchup_slots, -1).to(torch.int32)
            self.draft_inputs = tail, padded_positions, catchup_slots
        context_lens = (num_tokens + num_drafts_tensor).to(torch.int32)
        if len(set(num_drafts)) == 1 or pad:
            ragged = len(set(num_drafts)) > 1
            self.draft_layout = token_index.cuda(non_blocking=True), write_index.cuda(non_blocking=True), num_drafts_tensor if ragged else None
            if ragged:
                self.sampler_layout = self.draft_layout[0], num_drafts_tensor
            set_context(False, slot_mapping=slot_mapping.view(-1), context_lens=context_lens, block_tables=block_tables)
            return input_ids.view(-1), positions.view(-1)
        # Ragged: keep each sequence's first 1 + num_drafts rows, and a scratch row at the end that
        # propose writes the drafts of the sequences without them to.
        select = to_cuda([i * (k + 1) + j for i, num in enumerate(num_drafts) for j in range(1 + num)], torch.int64)    # rows of the (batch, k + 1) grid in use
        device = input_ids.device
        input_ids = torch.cat([input_ids.view(-1)[select], torch.zeros(1, dtype=torch.int64, device=device)])
        positions = torch.cat([positions.view(-1)[select], torch.zeros(1, dtype=torch.int64, device=device)])
        slot_mapping = torch.cat([slot_mapping.view(-1)[select], torch.full((1,), -1, dtype=torch.int32, device=device)])
        self.draft_layout = token_index.cuda(non_blocking=True), write_index.cuda(non_blocking=True), num_drafts_tensor
        cu_seqlens_q = [0]
        for num in num_drafts:
            cu_seqlens_q.append(cu_seqlens_q[-1] + 1 + num)
        self.set_ragged_context(seqs, num_drafts, cu_seqlens_q, slot_mapping[:-1], context_lens, block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @staticmethod
    def draft_temperatures(seqs: list[Sequence], temperatures: torch.Tensor | None) -> torch.Tensor | None:
        """The temperatures propose samples drafts at: None for a batch of greedy sequences, whose
        drafts are the argmax either way and whose rows the sampler verifies without q, so the
        step skips the (B, k, V) softmax and its storage (3% of a decode step at batch 32 on an
        A40 with Qwen3-8B + 0.6B, A40 job 244661)."""
        return temperatures if temperatures is not None and any(seq.temperature > 0 for seq in seqs) else None

    @torch.inference_mode()
    def forward(self, model: Qwen3ForCausalLM | Eagle3Head, graphs: dict | None, outputs: torch.Tensor | None, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool,
                hidden_states: torch.Tensor | None = None):
        """Hidden states of input_ids, through a captured CUDA graph when there is one for this
        decode batch size (none during warmup, or with enforce_eager). An EAGLE-3 head takes its
        rows' input hidden states too."""
        context = get_context()
        if is_prefill or graphs is None or context.context_lens.size(0) > 512:
            return model(input_ids, positions) if hidden_states is None else model(input_ids, positions, hidden_states)
        else:
            bs = context.context_lens.size(0)
            n = input_ids.size(0)    # bs * tokens per sequence
            graph = graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:n] = input_ids
            graph_vars["positions"][:n] = positions
            if hidden_states is not None:
                graph_vars["draft_hidden"][:n] = hidden_states
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:n] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return outputs[:n]

    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        graphs = outputs = None
        if self.graphs is not None and not is_prefill and get_context().cu_seqlens_q is None:    # a ragged batch runs eagerly
            q = input_ids.size(0) // get_context().context_lens.size(0)    # tokens per sequence in this step
            graphs, outputs = self.graphs[q], self.graph_vars["outputs"]
        return self.model.compute_logits(self.forward(self.model, graphs, outputs, input_ids, positions, is_prefill))

    @torch.inference_mode()
    def propose(self, input_ids: torch.Tensor, positions: torch.Tensor, temperatures: torch.Tensor | None = None) -> torch.Tensor:
        """Run the draft model k times and write its tokens into the draft columns of input_ids
        (one row of batch_size * (k + 1)). With speculative_sampled_drafts and the sequences'
        temperatures (rank 0), each draft is sampled from the draft's distribution at the
        sequence's temperature (DraftSampler) and the logits it came from, with their log-sum-exp,
        are kept in draft_logits and draft_log_norms for RejectionSampler, which reads q from
        them and accepts with min(1, p/q)
        (exact speculative sampling); otherwise the drafts are the draft's argmax, one-hot
        proposals the sampler accepts with p. The first step scores the
        tokens the draft KV cache lacks (see prepare_decode_with_draft_model) and proposes from the
        newest token's logits; each later step scores the draft just proposed.

        An EAGLE-3 head (models/eagle3.py) runs the same steps one position behind the target:
        its catch-up rows take the target's hidden states from the hidden cache and end at the
        position before the newest token, and the row that scores draft i sits where the target
        scores draft i - 1, taking the head's own output of the step before as its hidden state."""
        context = get_context()    # the target's
        context_lens, block_tables = context.context_lens, context.block_tables
        tokens, catchup_positions, catchup_slots = self.draft_inputs
        token_index, write_index, num_drafts = self.draft_layout    # num_drafts: None unless the batch is ragged
        if num_drafts is not None:
            num_drafts = num_drafts.to(torch.int32)    # the attention kernel wants int32 context lengths
        bs, m = tokens.shape
        k = token_index.size(1) - 1
        slot_mapping = context.slot_mapping
        sampling = self.sampled_drafts and temperatures is not None and self.rank == 0
        eagle = self.eagle
        shift = int(eagle)    # the head's rows sit one position behind the target's
        input_hidden_states = None    # a head's: the target's at the catch-up rows, then its own output
        self.draft_logits = self.draft_log_norms = None
        for i in range(k):
            if i == 0:
                q, step_positions, step_slots = m, catchup_positions, catchup_slots
                step_context_lens = (context_lens - k if num_drafts is None else context_lens - num_drafts) - shift
                if eagle:
                    input_hidden_states = self.draft_model.read_hidden_states(step_slots).view(bs * m, -1)
            else:    # the newest draft, at the position the target scores it
                rows = token_index[:, i - shift]
                q, step_positions, step_slots = 1, positions[rows], slot_mapping[rows]
                step_context_lens = context_lens - k + i - shift
                if num_drafts is not None:
                    # A sequence with fewer drafts scores a dummy row (its last one again, its KV
                    # masked) at its own context length, and its proposal goes to the scratch row.
                    step_slots = torch.where(num_drafts > i, step_slots, -1)
                    step_context_lens = context_lens - num_drafts + torch.minimum(num_drafts, torch.tensor(i, dtype=torch.int32, device=num_drafts.device)) - shift
            set_context(False, slot_mapping=step_slots.reshape(-1), context_lens=step_context_lens, block_tables=block_tables)
            # Graphs exist for 1 and 2 tokens per sequence (a head: 1 to K + 1); a longer catch-up, after steps without drafts, runs eagerly.
            graphs, outputs = (self.draft_graphs.get(q), self.graph_vars["draft_outputs"]) if self.draft_graphs is not None else (None, None)
            hidden_states = self.forward(self.draft_model, graphs, outputs, tokens.reshape(-1), step_positions.reshape(-1), False, input_hidden_states)
            last_rows = hidden_states[q - 1::q]
            if eagle:    # the normed output the logits come from, and the residual stream the next step feeds back
                last_rows, input_hidden_states = (t.contiguous() for t in last_rows.chunk(2, dim=-1))
            logits = self.draft_model.compute_logits(last_rows)
            if sampling:
                tokens, log_norms = self.draft_sampler(logits, temperatures)
                if i == 0:
                    self.draft_logits = torch.empty((bs, k) + logits.shape[1:], dtype=logits.dtype, device=logits.device)
                    self.draft_log_norms = torch.empty((bs, k), dtype=log_norms.dtype, device=logits.device)
                self.draft_logits[:, i] = logits
                self.draft_log_norms[:, i] = log_norms
            else:
                tokens = logits.argmax(dim=-1) if self.rank == 0 else torch.empty(bs, dtype=torch.int64, device="cuda")
            if self.world_size > 1:
                dist.broadcast(tokens, 0)
            input_ids[write_index[:, i]] = tokens
        restore_context(context)
        return input_ids

    def decode_logits(self, seqs: list[Sequence], is_prefill: bool, temperatures: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The step's inputs, drafts included, its logits, and its number of drafts (the most any
        sequence has: with speculative_per_sequence a batch can be ragged, see ragged_layout)."""
        k = 0 if is_prefill else max(seq.num_scheduled_tokens for seq in seqs) - 1
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        self.draft_logits = self.draft_log_norms = None
        if self.draft_model is not None and k:
            input_ids = self.propose(input_ids, positions, temperatures)
        n = input_ids.size(0) if get_context().cu_seqlens_q is None else sum(seq.num_scheduled_tokens for seq in seqs)    # a varlen ragged batch's inputs end with a scratch row
        logits = self.run_model(input_ids[:n], positions[:n], is_prefill)
        if self.draft_model is not None and is_prefill:    # fills the draft KV cache
            if self.eagle:
                self.prefill_head(seqs, input_ids, positions)
            else:
                self.forward(self.draft_model, None, None, input_ids, positions, True)
        return input_ids, logits, k

    def prefill_head(self, seqs: list[Sequence], input_ids: torch.Tensor, positions: torch.Tensor):
        """The EAGLE-3 head's prefill, on the target's context: its row at position p takes the
        target's hidden state at p (just written to the hidden cache) and the token after p, so
        the input ids shift left by one within each sequence, and a chunk's last row takes the
        prompt's next token, or 0 when the prompt ends there (the first decode step scores that
        row again; see Scheduler.commit)."""
        head = self.draft_model
        context = get_context()
        if head.hidden_cache.numel():
            hidden_states = head.read_hidden_states(context.slot_mapping)
            next_tokens = [seq.token_ids[end] if (end := seq.num_cached_tokens + seq.num_scheduled_tokens) < seq.num_tokens else 0 for seq in seqs]
            next_tokens = torch.tensor(next_tokens, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            input_ids = input_ids.roll(-1)
            input_ids[(context.cu_seqlens_q[1:] - 1).long()] = next_tokens
        else:    # warmup: no caches yet, the forward is only compiled
            hidden_states = torch.zeros(input_ids.size(0), head.config.hidden_size, dtype=head.config.dtype, device=input_ids.device)
        self.forward(head, None, None, input_ids, positions, True, hidden_states)

    def sample(self, seqs: list[Sequence], input_ids: torch.Tensor, logits: torch.Tensor, temperatures: torch.Tensor, k: int) -> torch.Tensor:
        """The step's sampled tokens: one per sequence without drafts, else RejectionSampler's row
        per sequence (its accepted drafts, one sampled token, -1 padding)."""
        if not k:
            return self.sampler(logits, temperatures)
        if self.sampler_layout is None:
            return self.rejection_sampler(logits, temperatures, input_ids[:len(seqs) * (k + 1)].view(len(seqs), k + 1)[:, 1:], draft_logits=self.draft_logits, draft_log_norms=self.draft_log_norms)
        token_index, num_drafts = self.sampler_layout
        return self.rejection_sampler(logits, temperatures, input_ids[token_index[:, 1:]], token_index, num_drafts, self.draft_logits, self.draft_log_norms)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | list[list[int]]:
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        input_ids, logits, k = self.decode_logits(seqs, is_prefill, self.draft_temperatures(seqs, temperatures))
        token_ids = None
        if self.rank == 0:
            token_ids = self.sample(seqs, input_ids, logits, temperatures, k).tolist()
            if k:
                token_ids = accepted_token_ids(token_ids)
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_async(self, seqs: list[Sequence], is_prefill: bool) -> "AsyncOutput | None":
        """Like run, but returns before the GPU finishes. The sampled tokens stay on the GPU as
        inputs for the next decode step, and are copied to the host in the background."""
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        input_ids, logits, k = self.decode_logits(seqs, is_prefill, self.draft_temperatures(seqs, temperatures))
        if self.rank == 0:
            token_ids = self.sample(seqs, input_ids, logits, temperatures, k)
        else:
            token_ids = torch.empty((len(seqs), k + 1) if k else len(seqs), dtype=torch.int64, device="cuda")
        if self.world_size > 1:
            dist.broadcast(token_ids, 0)    # every rank needs them as next-step inputs
        # With speculative decoding every step's output has a row per sequence, padded to the
        # configured number of drafts (see AsyncOutput), whether or not this step drafted all of them.
        K = self.num_speculative_tokens
        if K and token_ids.dim() == 1:
            self.last_token_ids = torch.nn.functional.pad(token_ids.unsqueeze(1), (0, K), value=-1)
        elif K and token_ids.size(1) < K + 1:
            self.last_token_ids = torch.nn.functional.pad(token_ids, (0, K + 1 - token_ids.size(1)), value=-1)
        else:
            self.last_token_ids = token_ids
        reset_context()
        return AsyncOutput(token_ids) if self.rank == 0 else None

    @torch.inference_mode()
    def capture_graphs(self, model: Qwen3ForCausalLM | Eagle3Head, outputs: torch.Tensor, q: int, with_hidden_states: bool = False) -> dict[int, torch.cuda.CUDAGraph]:
        """One graph per batch size for decode steps of q tokens per sequence. All graphs share
        the input buffers in graph_vars and the memory pool; they are replayed one at a time."""
        graph_vars = self.graph_vars
        input_ids, positions = graph_vars["input_ids"], graph_vars["positions"]
        graphs = {}
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            n = bs * q
            inputs = (input_ids[:n], positions[:n]) + ((graph_vars["draft_hidden"][:n],) if with_hidden_states else ())
            set_context(False, slot_mapping=graph_vars["slot_mapping"][:n], context_lens=graph_vars["context_lens"][:bs], block_tables=graph_vars["block_tables"][:bs])
            outputs[:n] = model(*inputs)    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:n] = model(*inputs)    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        return graphs

    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        q = 1 + self.num_speculative_tokens    # tokens per sequence in a decode step with drafts
        max_num_blocks = (config.max_model_len + q - 1 + self.block_size - 1) // self.block_size
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_pool = None
        self.graph_vars = dict(
            input_ids=torch.zeros(max_bs * q, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(max_bs * q, dtype=torch.int64, device="cuda"),
            slot_mapping=torch.zeros(max_bs * q, dtype=torch.int32, device="cuda"),
            context_lens=torch.zeros(max_bs, dtype=torch.int32, device="cuda"),
            block_tables=torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cuda"),
            outputs=torch.zeros(max_bs * q, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda"),
        )
        # Decode steps without drafts (a large batch) or with fewer of them (adaptive k, see
        # Scheduler.schedule) need their own graphs.
        self.graphs = {}
        for q in self.decode_query_lengths():
            self.graphs[q] = self.capture_graphs(self.model, self.graph_vars["outputs"], q)
        self.draft_graphs = {}
        if self.draft_model is not None and self.eagle:
            # The head's catch-up step scores as many rows per sequence as the step before
            # accepted, 1 to K + 1 (more after steps without drafts: eager), the others one.
            hidden_size = config.draft_hf_config.hidden_size
            self.graph_vars["draft_hidden"] = torch.zeros(max_bs * q, hidden_size, dtype=config.draft_hf_config.dtype, device="cuda")
            self.graph_vars["draft_outputs"] = torch.zeros(max_bs * q, 2 * hidden_size, dtype=config.draft_hf_config.dtype, device="cuda")
            for rows in range(q, 0, -1):
                self.draft_graphs[rows] = self.capture_graphs(self.draft_model, self.graph_vars["draft_outputs"], rows, with_hidden_states=True)
        elif self.draft_model is not None:
            # The first draft step scores two tokens per sequence, the others one (see propose).
            self.graph_vars["draft_outputs"] = torch.zeros(max_bs * 2, config.draft_hf_config.hidden_size, dtype=config.draft_hf_config.dtype, device="cuda")
            for q in (2, 1):
                self.draft_graphs[q] = self.capture_graphs(self.draft_model, self.graph_vars["draft_outputs"], q)
