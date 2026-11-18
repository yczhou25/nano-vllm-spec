"""The EAGLE-3 head and the target sharing one token embedding (Eagle3Head.share_embeddings),
on the real modules, built small on the CPU."""
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.models import eagle3
from nanovllm.models.eagle3 import Eagle3Head
from nanovllm.utils.loader import load_model

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
    assert head.embed_scale == 1.0 and torch.equal(head.embed(ids), before)
    # One table: the head's copy is gone as soon as it shares (no garbage collection in between;
    # the engine sizes its KV cache from what is free right after), and its parameters list the
    # target's weight under the checkpoint's name.
    assert own_weight() is None
    assert dict(head.named_parameters())["embed_tokens.weight"] is embed_tokens.weight
    embed_tokens.weight.data[5] += 1
    assert torch.equal(head.embed_tokens(ids)[1], before[1] + 1)


def test_head_keeps_an_embedding_that_differs_in_one_weight(head):
    embed_tokens = target_embedding(head)
    embed_tokens.weight.data[VOCAB - 1, HIDDEN - 1] += 2.0    # a tenth of the row's length: no rounding
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


def test_head_without_a_table_of_its_own_takes_the_target_embedding(head):
    embed_tokens = target_embedding(head)
    embed_tokens.weight.data += 0.125    # nothing like what the head's never-loaded table happens to hold
    own_weight = weakref.ref(head.embed_tokens.weight)

    assert head.share_embeddings(embed_tokens, own_loaded=False)

    assert head.embed_tokens is embed_tokens
    assert own_weight() is None


def test_head_without_a_table_of_its_own_needs_the_target_shape(head):
    with pytest.raises(AssertionError, match="no table of its own"):
        head.share_embeddings(target_embedding(head, VOCAB // 2), own_loaded=False)


@pytest.mark.parametrize("with_embedding", [True, False])
def test_loader_reports_whether_the_checkpoint_had_the_embedding(head, tmp_path, with_embedding):
    target = target_embedding(head)
    target.weight.data += 0.125
    checkpoint = {"fc.weight": torch.full_like(head.fc.weight, 0.5), "t2d": torch.ones(VOCAB, dtype=torch.bool)}
    if with_embedding:
        checkpoint["embed_tokens.weight"] = torch.full((VOCAB, HIDDEN), 0.25)
    save_file(checkpoint, str(tmp_path / "model.safetensors"))

    loaded = load_model(head, str(tmp_path))

    assert loaded == set(checkpoint)
    assert torch.equal(head.fc.weight, checkpoint["fc.weight"]) and head.t2d.all()
    # The head's own (different) table stays when the checkpoint brought one; without one the
    # head has only uninitialized memory there and embeds with the target's.
    assert head.share_embeddings(target, "embed_tokens.weight" in loaded) == (not with_embedding)
    expected = torch.full((HIDDEN,), 0.25) if with_embedding else target.weight[3]
    assert torch.equal(head.embed_tokens(torch.tensor([3]))[0], expected)


SCALE = 0.876864    # what RedHatAI/Qwen3-8B-speculator.eagle3's table is of Qwen3-8B's


def scaled_copy(weight, scale=SCALE, noise=0.002):
    """`weight` times `scale`, every element off by up to `noise` of its value, as rounding to bf16 leaves a table."""
    generator = torch.Generator().manual_seed(1)
    return weight * scale * (1 + noise * (2 * torch.rand(weight.shape, generator=generator) - 1))


def test_head_takes_the_target_embedding_for_a_scaled_copy(head):
    embed_tokens = target_embedding(head)
    embed_tokens.weight.data[9] = 0    # an unused token's row
    own = scaled_copy(embed_tokens.weight.data)
    head.embed_tokens.weight.data.copy_(own)
    own_weight = weakref.ref(head.embed_tokens.weight)
    ids = torch.tensor([0, 5, 9, VOCAB - 1])

    assert head.share_embeddings(embed_tokens)

    assert head.embed_tokens is embed_tokens and own_weight() is None
    assert head.embed_scale == pytest.approx(SCALE, rel=1e-4)
    # The target's rows times the factor are the head's rows again, to the rounding.
    embeds = head.embed(ids)
    assert embeds.dtype == own.dtype
    assert ((embeds - own[ids]).norm(dim=1) <= 0.002 * own[ids].norm(dim=1)).all()


@pytest.mark.parametrize("change", ["row_scaled_otherwise", "row_turned", "noisy", "negated"])
def test_head_keeps_a_table_that_is_not_one_factor_times_the_target(head, change):
    embed_tokens = target_embedding(head)
    own = scaled_copy(embed_tokens.weight.data)
    if change == "row_scaled_otherwise":
        own[7] *= 1.02
    elif change == "row_turned":
        own[7, 0] += 0.03 * own[7].norm()
    elif change == "noisy":
        own = scaled_copy(embed_tokens.weight.data, noise=0.03)
    else:
        own = -own
    head.embed_tokens.weight.data.copy_(own)
    kept = head.embed_tokens

    assert not head.share_embeddings(embed_tokens)

    assert head.embed_tokens is kept and head.embed_scale == 1.0
    assert torch.equal(head.embed(torch.tensor([7]))[0], own[7])


def test_scale_of_copy_checks_every_row_against_the_tolerance():
    torch.manual_seed(2)
    target = torch.randn(VOCAB, HIDDEN)
    target[:16] = 0    # a first step of unused rows: the factor is the whole table's
    own = target * SCALE
    assert Eagle3Head.scale_of_copy(own, target, 1e-5, rows_per_step=16) == pytest.approx(SCALE, rel=1e-6)
    own[VOCAB - 1] *= 1.003    # the last row of the last step, 0.3% longer than the factor makes it
    assert Eagle3Head.scale_of_copy(own, target, 0.005, rows_per_step=16) == pytest.approx(SCALE, rel=1e-4)
    assert Eagle3Head.scale_of_copy(own, target, 0.002, rows_per_step=16) is None
    assert Eagle3Head.scale_of_copy(own[:16], target, 0.005) is None    # another shape
    assert Eagle3Head.scale_of_copy(own, torch.zeros_like(target), 0.005) is None

