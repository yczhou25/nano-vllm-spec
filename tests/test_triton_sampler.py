"""fused_sample, the Triton kernels behind Sampler and DraftSampler on the GPU, against the compiled
torch versions (torch_forward) and the distributions they sample from. Without a GPU the kernels
run in Triton's interpreter (TRITON_INTERPRET=1, set by conftest.py), about 10 ms per program, so
the cases here keep rows * tiles in the hundreds; small blocks give several tiles per row."""
import pytest
import torch

from nanovllm.layers.sampler import DraftSampler, Sampler, fused_sample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rows_of(batch_size, vocab_size, dtype=torch.float32):
    torch.manual_seed(batch_size * 1000 + vocab_size)
    return (torch.randn(batch_size, vocab_size) * 3).to(dtype).to(DEVICE)


@pytest.mark.parametrize("vocab_size, block", [(40, 16), (37, 8), (1024, 1024), (1187, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_greedy_tokens_and_log_norms_match_torch(vocab_size, block, dtype):
    logits = rows_of(12, vocab_size, dtype)
    logits[1, 5] = logits[1, 9] = logits[1].max() + 1      # a tie inside one tile
    logits[2, 0] = logits[2, -1] = logits[2].max() + 1     # a tie across tiles: the first index wins
    logits[3, :3] = float("-inf")                          # masked entries
    logits[6] = torch.arange(vocab_size, device=DEVICE) * 1e-12    # T=0 must take the greedy branch: at T=1e-10 these are 0..3.9 and the Gumbel noise would decide
    temperatures = torch.tensor([0.0, 0.0, 0.0, 0.7, 1.0, 0.3, 0.0, 1.5, 0.8, 2.0, 0.1, 1e-3], device=DEVICE)
    tokens, log_norms = fused_sample(logits, temperatures, block=block)
    greedy = logits.argmax(dim=-1)
    assert torch.equal(tokens[temperatures == 0], greedy[temperatures == 0])
    expected = torch.logsumexp(logits.float() / temperatures.clamp_min(1e-10).unsqueeze(1), dim=-1)
    assert torch.allclose(log_norms, expected, rtol=1e-5, atol=1e-4)
    assert tokens.dtype == torch.int64 and log_norms.dtype == torch.float32
    assert (tokens >= 0).all() and (tokens < vocab_size).all()
    assert torch.equal(fused_sample(logits, torch.zeros_like(temperatures), block=block)[0], greedy)


@pytest.mark.filterwarnings("ignore:divide by zero")    # the interpreter's numpy log(0) for the -inf row
def test_a_row_of_minus_inf_gives_token_0_and_an_infinite_log_norm():
    logits = rows_of(2, 40)
    logits[0] = float("-inf")
    tokens, log_norms = fused_sample(logits, torch.tensor([0.9, 0.9], device=DEVICE), block=16)
    assert tokens[0] == 0 and log_norms[0] == float("-inf")
    assert log_norms[1] != float("-inf")


def test_samples_follow_softmax_across_tiles():
    # One peaked row, repeated: its sample frequencies against softmax(logits / T).
    vocab_size, temperature, rows = 40, 0.8, 500
    torch.manual_seed(3)
    row = torch.randn(vocab_size) * 2
    logits = row.expand(rows, vocab_size).contiguous().to(DEVICE)
    torch.manual_seed(1)
    tokens, _ = fused_sample(logits, torch.full((rows,), temperature, device=DEVICE), block=16)
    probs = torch.softmax(row / temperature, dim=-1)
    counts = torch.bincount(tokens.cpu(), minlength=vocab_size).float() / rows
    standard_error = (probs * (1 - probs) / rows).sqrt()
    assert ((counts - probs).abs() <= 4.5 * standard_error + 1e-3).all(), (counts - probs).abs().max()
    assert (tokens.cpu() != row.argmax()).any()    # the top token is not the only one drawn


def test_equal_logits_sample_uniformly():
    # Three tiles per row: if two tiles drew the same noise, the first of them would always win.
    rows, vocab_size = 300, 24
    torch.manual_seed(2)
    tokens, log_norms = fused_sample(torch.zeros(rows, vocab_size, device=DEVICE), torch.ones(rows, device=DEVICE), block=8)
    counts = torch.bincount(tokens.cpu(), minlength=vocab_size).float() / rows
    assert (counts - 1 / vocab_size).abs().max() < 4.5 * (1 / vocab_size * (1 - 1 / vocab_size) / rows) ** 0.5
    assert torch.allclose(log_norms, torch.full((rows,), float(torch.log(torch.tensor(vocab_size))), device=DEVICE))


def test_manual_seed_fixes_the_samples_and_calls_differ():
    logits = rows_of(8, 40)
    temperatures = torch.ones(8, device=DEVICE)
    torch.manual_seed(5)
    first = fused_sample(logits, temperatures, block=16)[0]
    second = fused_sample(logits, temperatures, block=16)[0]
    torch.manual_seed(5)
    again = fused_sample(logits, temperatures, block=16)[0]
    assert torch.equal(first, again)
    assert not torch.equal(first, second)


def test_samplers_dispatch_to_the_kernels_on_the_gpu_and_torch_elsewhere():
    logits = rows_of(6, 64, torch.bfloat16)
    temperatures = torch.tensor([0.0, 0.5, 0.0, 1.0, 0.0, 0.8], device=DEVICE)
    tokens, log_norms = DraftSampler()(logits, temperatures)
    expected_tokens, torch_norms = DraftSampler().torch_forward(logits, temperatures)
    assert torch.equal(tokens[temperatures == 0], expected_tokens[temperatures == 0])
    # Both paths use approximate exp/log/division on the GPU (Triton's div.full and ex2/lg2.approx,
    # Inductor's libdevice); compare each with a float64 reference rather than with each other. Both
    # are within 1e-7 relative (bench_sampler.py --check, A40 job 1275031), except that Inductor's
    # compiled log-sum-exp overflows to inf on the T=0 rows (logits / 1e-10 ~ 1e11), where no log-norm
    # is ever read: the kernel's value there is finite and checked, the torch path's is not.
    reference = torch.logsumexp(logits.double() / temperatures.double().clamp_min(1e-10).unsqueeze(1), dim=-1)
    assert torch.allclose(log_norms.double(), reference, rtol=1e-5, atol=1e-3), (log_norms - reference).abs().max()
    sampled = temperatures > 0
    assert torch.allclose(torch_norms.double()[sampled], reference[sampled], rtol=1e-5, atol=1e-3), (torch_norms - reference).abs()[sampled].max()
    sampled = Sampler()(logits, temperatures)
    assert sampled.shape == (6,) and sampled.dtype == torch.int64
    assert torch.equal(sampled[temperatures == 0], logits.argmax(dim=-1)[temperatures == 0])
