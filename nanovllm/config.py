import json
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    async_scheduling: bool = False
    num_speculative_tokens: int = 0
    speculative_model: str | None = None    # a smaller model that drafts the speculative tokens, or an EAGLE-3 head (speculators format) for the target; n-gram lookup if None
    speculative_method: str | None = None   # what speculative_model is, read from its config: "model" or "eagle3"
    speculative_disable_by_batch_size: int | None = None    # decode without drafts while more sequences than this are running
    speculative_min_acceptance: float | None = None    # decode without drafts while the measured draft acceptance rate is below this
    speculative_per_sequence: bool = False    # with speculative_min_acceptance: a number of drafts per sequence, from its own acceptance rates
    speculative_sampled_drafts: bool = True    # with a draft model: each draft is sampled from the draft's distribution at the sequence's temperature and verified with min(1, p/q) (exact speculative sampling); False: the draft's argmax, as the n-gram proposer (one-hot q)
    speculative_ragged: str = "pad"    # how a batch with different numbers of drafts runs: "pad" (every sequence padded to the most drafts, fixed-width attention and CUDA graphs; faster except on a mixed batch of 32) or "varlen" (a row per token, varlen attention, no CUDA graph)
    hf_config: AutoConfig | None = None
    draft_hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.num_speculative_tokens >= 0
        assert not (self.async_scheduling and self.num_speculative_tokens and self.speculative_model is None), \
            "n-gram drafts need the sampled tokens on the CPU; with async scheduling use a draft model (speculative_model)"
        assert self.speculative_disable_by_batch_size is None or (self.num_speculative_tokens and self.speculative_disable_by_batch_size >= 1)
        assert self.speculative_min_acceptance is None or (self.num_speculative_tokens and 0.0 <= self.speculative_min_acceptance <= 1.0)
        assert not self.speculative_per_sequence or self.speculative_min_acceptance is not None, "speculative_per_sequence needs speculative_min_acceptance"
        assert self.speculative_ragged in ("varlen", "pad")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculative_model is not None:
            assert os.path.isdir(self.speculative_model)
            assert self.num_speculative_tokens > 0, "speculative_model needs num_speculative_tokens"
            with open(os.path.join(self.speculative_model, "config.json")) as f:
                speculative_config = json.load(f)
            if speculative_config.get("speculators_model_type") == "eagle3":
                from nanovllm.models.eagle3 import Eagle3Config
                self.speculative_method = "eagle3"
                self.draft_hf_config = Eagle3Config(self.speculative_model, self.hf_config)
                assert self.draft_hf_config.target_hidden_size == self.hf_config.hidden_size, "the EAGLE-3 head was trained for a target of another width"
            else:
                self.speculative_method = "model"
                self.draft_hf_config = AutoConfig.from_pretrained(self.speculative_model)
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size, "the draft model must share the target's tokenizer"
            self.max_model_len = min(self.max_model_len, self.draft_hf_config.max_position_embeddings)
