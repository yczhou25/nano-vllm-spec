"""The EAGLE-3 head and the target sharing one token embedding (Eagle3Head.share_embeddings),
on the real modules, built small on the CPU."""
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.models import eagle3
from nanovllm.models.eagle3 import Eagle3Head

VOCAB, HIDDEN = 64, 16


@pytest.fixture
def head(monkeypatch):
    # The layers read their tensor-parallel rank from the process group, and the head's attention
    # needs the GPU kernels: a single-process group and no attention will do for the embedding.
    own_group = not dist.is_initialized()
    if own_group:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    monkeypatch.setattr(eagle3, "Attention", lambda *args: None)
    config = SimpleNamespace(vocab_size=VOCAB, hidden_size=HIDDEN, target_hidden_size=HIDDEN, aux_hidden_state_layers=(0, 1, 2),
                             intermediate_size=2 * HIDDEN, num_attention_heads=4, num_key_value_heads=2, head_dim=4, attention_bias=False,
                             hidden_act="silu", rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=32,
                             draft_vocab_size=8, norm_before_residual=True)
    head = Eagle3Head(config)
    torch.manual_seed(0)
    # Multiples of 1/8: exact in any float type, so only the dtype itself tells two tables apart.
    head.embed_tokens.weight.data.copy_(torch.randint(-64, 64, (VOCAB, HIDDEN)) / 8)
    yield head
    if own_group:
        dist.destroy_process_group()


def target_embedding(head, vocab=VOCAB, dtype=None):
    """A target's embedding module holding the head's weights (their first `vocab` rows, as `dtype`)."""
    weight = head.embed_tokens.weight.data[:vocab]
    embed_tokens = VocabParallelEmbedding(vocab, HIDDEN)
    embed_tokens.weight.data = weight.to(dtype or weight.dtype).clone()
    return embed_tokens


def test_head_takes_the_target_embedding_when_the_weights_are_the_same(head):
    embed_tokens = target_embedding(head)
    own_weight = weakref.ref(head.embed_tokens.weight)
    ids = torch.tensor([0, 5, VOCAB - 1])
    before = head.embed_tokens(ids).detach()    # not its graph, which holds the weight

    assert head.share_embeddings(embed_tokens)

    assert head.embed_tokens is embed_tokens
    assert torch.equal(head.embed_tokens(ids), before)
    # One table: the head's copy is gone as soon as it shares (no garbage collection in between;
    # the engine sizes its KV cache from what is free right after), and its parameters list the
    # target's weight under the checkpoint's name.
    assert own_weight() is None
    assert dict(head.named_parameters())["embed_tokens.weight"] is embed_tokens.weight
    embed_tokens.weight.data[5] += 1
    assert torch.equal(head.embed_tokens(ids)[1], before[1] + 1)


def test_head_keeps_an_embedding_that_differs_in_one_weight(head):
    embed_tokens = target_embedding(head)
    embed_tokens.weight.data[VOCAB - 1, HIDDEN - 1] += 0.125
    own = head.embed_tokens

    assert not head.share_embeddings(embed_tokens)

    assert head.embed_tokens is own
    assert own.weight.weight_loader == own.weight_loader    # and can still be loaded


@pytest.mark.parametrize("vocab, dtype", [(VOCAB // 2, None), (VOCAB, torch.float64)], ids=["smaller_vocab", "other_dtype"])
def test_head_keeps_an_embedding_of_another_shape_or_dtype(head, vocab, dtype):
    embed_tokens = target_embedding(head, vocab, dtype)
    own = head.embed_tokens

    assert not head.share_embeddings(embed_tokens)

    assert head.embed_tokens is own
    assert own.weight.dtype == torch.get_default_dtype()
