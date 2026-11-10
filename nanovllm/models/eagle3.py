"""An EAGLE-3 draft head (speculators checkpoint format, as vLLM loads it): one Llama-style
decoder layer that drafts the target's next tokens from the target's own hidden states.

The head's row at position p takes the target's fused hidden state at p (three of the target's
layers, concatenated and projected by `fc`) and the embedding of token t[p+1], and predicts
t[p+2]: one position behind the target, which at p takes t[p] and predicts t[p+1]. Later drafts
feed the head's own output (the residual stream before `norm`) back as the hidden state.

Its logits cover a draft vocabulary (the target's most frequent tokens); compute_logits maps
them into the target's vocabulary with `d2t`, so the samplers see the target's vocabulary as
with a draft model. See docs/spec-decode-design.md, "EAGLE-3 草稿头".
"""
import json
import os
from typing import ClassVar

import torch
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3MLP
from nanovllm.utils.context import get_context


class Eagle3Config:
    """The head's config.json, read into the attributes Config and ModelRunner expect of a
    model config (hidden sizes, heads, dtype, a single hidden layer for the KV cache)."""

    def __init__(self, path: str, target_config):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        assert config.get("speculators_model_type") == "eagle3", f"{path} is not an EAGLE-3 head"
        layer = config["transformer_layer_config"]
        self.hidden_size = layer["hidden_size"]
        self.intermediate_size = layer["intermediate_size"]
        self.num_attention_heads = layer["num_attention_heads"]
        self.num_key_value_heads = layer.get("num_key_value_heads", self.num_attention_heads)
        self.head_dim = layer.get("head_dim") or self.hidden_size // self.num_attention_heads
        self.rms_norm_eps = layer.get("rms_norm_eps", 1e-6)
        self.rope_theta = layer.get("rope_theta", 10000.0)
        self.attention_bias = layer.get("attention_bias", False)
        self.hidden_act = layer.get("hidden_act", "silu")
        self.max_position_embeddings = layer.get("max_position_embeddings", target_config.max_position_embeddings)
        self.num_hidden_layers = 1
        self.vocab_size = layer.get("vocab_size", target_config.vocab_size)    # the target's: the logits come out mapped to it
        self.draft_vocab_size = config.get("draft_vocab_size", 32000)
        self.norm_before_residual = config.get("norm_before_residual", False)
        self.target_hidden_size = config.get("target_hidden_size") or target_config.hidden_size
        num_layers = target_config.num_hidden_layers
        # The target layers whose inputs the head reads: vLLM's default for EAGLE-3 unless the head says.
        self.aux_hidden_state_layers = tuple(config.get("eagle_aux_hidden_state_layer_ids") or (2, num_layers // 2, num_layers - 3))
        self.dtype = getattr(torch, config.get("torch_dtype", "bfloat16"))
        self.tie_word_embeddings = False


class Eagle3Attention(nn.Module):

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(2 * config.hidden_size, self.head_dim, self.num_heads, self.num_kv_heads, bias=config.attention_bias)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=config.max_position_embeddings, base=config.rope_theta)
        self.attn = Attention(self.qkv_proj.num_heads, self.head_dim, self.head_dim ** -0.5, self.qkv_proj.num_kv_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_size, kv_size = self.qkv_proj.num_heads * self.head_dim, self.qkv_proj.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(-1, self.qkv_proj.num_heads, self.head_dim)
        k = k.view(-1, self.qkv_proj.num_kv_heads, self.head_dim)
        v = v.view(-1, self.qkv_proj.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Eagle3DecoderLayer(nn.Module):

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.norm_before_residual = config.norm_before_residual
        self.self_attn = Eagle3Attention(config)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, embeds: torch.Tensor, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = self.input_layernorm(embeds)
        if self.norm_before_residual:
            hidden_states = self.hidden_norm(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
        hidden_states = self.self_attn(positions, torch.cat([embeds, hidden_states], dim=-1))
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Eagle3Head(nn.Module):
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.fc = ReplicatedLinear(len(config.aux_hidden_state_layers) * config.target_hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([Eagle3DecoderLayer(config)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size)
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))
        self.target_token_ids = None    # d2t as the target token of every draft token, once the weights are loaded
        # The target's fused hidden state of every position its KV cache holds, at the same slot
        # (ModelRunner binds it: num_kvcache_blocks * block_size rows, plus one that masked slots
        # write to). The target writes it every forward (store_hidden_states); the head's catch-up
        # step reads the positions it lacks (ModelRunner.propose).
        self.hidden_cache = torch.tensor([])

    def store_hidden_states(self, aux_hidden_states: torch.Tensor):
        """Called by the target model with its aux layers' inputs, concatenated, (N, 3 * target
        hidden): fuses them and writes the result to the hidden cache at the context's slots."""
        if not self.hidden_cache.numel():    # warmup, before the caches are allocated
            return
        fused = self.fc(aux_hidden_states)
        slots = get_context().slot_mapping.long()
        self.hidden_cache[torch.where(slots >= 0, slots, self.hidden_cache.size(0) - 1)] = fused

    def read_hidden_states(self, slots: torch.Tensor) -> torch.Tensor:
        """The fused hidden states at slots (-1: whatever the spare row holds; such a row's output is not used)."""
        return self.hidden_cache[torch.where(slots >= 0, slots, self.hidden_cache.size(0) - 1).long()]

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """(N, 2 * hidden): the normed output compute_logits takes, then the residual stream the
        next draft step feeds back as its hidden state."""
        embeds = self.embed_tokens(input_ids)
        hidden_states, residual = self.layers[0](positions, embeds, hidden_states)
        normed, residual = self.norm(hidden_states, residual)
        return torch.cat([normed, residual], dim=-1)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Logits over the target's vocabulary: the draft vocabulary's at their target tokens, -inf elsewhere."""
        logits = self.lm_head(hidden_states)
        if logits is None:    # a rank other than 0
            return None
        if self.target_token_ids is None:
            self.target_token_ids = torch.arange(self.d2t.numel(), device=self.d2t.device) + self.d2t
        mapped = logits.new_full((logits.size(0), self.config.vocab_size), float("-inf"))
        mapped[:, self.target_token_ids] = logits
        return mapped
