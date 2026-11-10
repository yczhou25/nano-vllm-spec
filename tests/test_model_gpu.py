"""Greedy outputs checked token by token against Hugging Face transformers on a real model.

nano-vllm runs in bf16 and the reference in fp32, so near-ties can legitimately flip. Instead of
comparing two free-running generations, the reference scores nano-vllm's own output with teacher
forcing, and every chosen token must be within LOGIT_TOL of the reference's best logit.
"""
import atexit
import os

import pytest
import torch

from nanovllm.engine.sequence import Sequence

pytestmark = pytest.mark.gpu

MODEL = os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B/"))
# The draft model defaults to the target itself, whose greedy drafts should nearly all be accepted.
DRAFT_MODEL = os.path.expanduser(os.environ.get("NANOVLLM_TEST_DRAFT_MODEL", MODEL))
# An EAGLE-3 head for the test model (none: its modes are skipped). The head's rows only affect
# the acceptance rate, so these modes are the same checks as the draft model's.
EAGLE_MODEL = os.environ.get("NANOVLLM_TEST_EAGLE_MODEL")
LOGIT_TOL = float(os.environ.get("NANOVLLM_TEST_LOGIT_TOL", "0.5"))
# A large test model needs the GPU shared with the reference: less of it for the engine, and the reference in bf16.
GPU_MEMORY_UTILIZATION = float(os.environ.get("NANOVLLM_TEST_GPU_MEMORY_UTILIZATION", "0.5"))
REFERENCE_DTYPE = getattr(torch, os.environ.get("NANOVLLM_TEST_REFERENCE_DTYPE", "float32"))
MAX_TOKENS = 64

QUESTIONS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "What is the capital of France?",
    "Write a haiku about GPUs.",
]
DOCUMENT = " ".join(f"Paragraph {i}: the scheduler batches requests, the block manager hands out KV cache blocks, "
                    f"and the model runner turns them into input tensors for step {i}." for i in range(60))
DOCUMENT_QUESTIONS = ["Summarize the document in one sentence.", "Which component hands out KV cache blocks?"]


ENGINE_OPTIONS = {
    "sync": {},
    "async": {"async_scheduling": True},
    "spec": {"num_speculative_tokens": 4},
    "draft": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL},
    "async_draft": {"async_scheduling": True, "num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL},
    # Drafts only while at most 2 sequences run: the 4 questions decode without them, the 2 documents with them.
    "draft_adaptive": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_disable_by_batch_size": 2},
    "draft_greedy_drafts": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_sampled_drafts": False},
    # Acceptance switch: n-gram drafts of the questions are mostly rejected, so decode runs mostly without them,
    # with a probe step now and then; the draft model's are mostly accepted (always, when it is the target itself).
    "spec_acceptance": {"num_speculative_tokens": 4, "speculative_min_acceptance": 0.5},
    "draft_acceptance": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_min_acceptance": 0.5},
    # The same per sequence: the questions' n-gram drafts get cut at different positions, so decode steps are ragged
    # (varlen attention, no CUDA graph); the draft model's are mostly accepted, with the odd near-tie flipped by bf16 noise.
    "spec_per_sequence": {"num_speculative_tokens": 4, "speculative_min_acceptance": 0.5, "speculative_per_sequence": True},
    "draft_per_sequence": {"num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_min_acceptance": 0.9, "speculative_per_sequence": True},
    "async_draft_per_sequence": {"async_scheduling": True, "num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_min_acceptance": 0.9, "speculative_per_sequence": True},
    # The same with ragged batches padded to the fixed width (dummy rows, CUDA graphs) instead of varlen attention.
    "spec_per_sequence_varlen": {"num_speculative_tokens": 4, "speculative_min_acceptance": 0.5, "speculative_per_sequence": True, "speculative_ragged": "varlen"},
    "async_draft_per_sequence_varlen": {"async_scheduling": True, "num_speculative_tokens": 4, "speculative_model": DRAFT_MODEL, "speculative_min_acceptance": 0.9, "speculative_per_sequence": True, "speculative_ragged": "varlen"},
    # An EAGLE-3 head drafting for the target (NANOVLLM_TEST_EAGLE_MODEL), with and without async scheduling and adaptive k.
    "eagle": {"num_speculative_tokens": 4, "speculative_model": EAGLE_MODEL},
    "async_eagle": {"async_scheduling": True, "num_speculative_tokens": 4, "speculative_model": EAGLE_MODEL},
    "eagle_acceptance": {"num_speculative_tokens": 4, "speculative_model": EAGLE_MODEL, "speculative_min_acceptance": 0.5},
}


@pytest.fixture(scope="module", params=list(ENGINE_OPTIONS))
def llm(request):
    if not os.path.isdir(MODEL):
        pytest.skip(f"no model at {MODEL}; set NANOVLLM_TEST_MODEL")
    if "eagle" in request.param and not EAGLE_MODEL:
        pytest.skip("no EAGLE-3 head; set NANOVLLM_TEST_EAGLE_MODEL")
    from nanovllm import LLM
    # A small token budget sends the long prompts through chunked prefill.
    engine = LLM(MODEL, max_model_len=4096, max_num_batched_tokens=1024, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                 **ENGINE_OPTIONS[request.param])
    yield engine
    engine.exit()
    atexit.unregister(engine.exit)


@pytest.fixture(scope="module")
def reference(llm):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=REFERENCE_DTYPE).cuda().eval()


def chat(tokenizer, text, thinking=True):
    # Qwen3 thinks before it answers unless the template is told not to (bench_spec.py turns it off).
    return tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking))


@torch.inference_mode()
def check(reference, prompt_ids, completion_ids):
    ids = torch.tensor([prompt_ids + completion_ids], device="cuda")
    logits = reference(ids).logits[0, len(prompt_ids) - 1:-1].float()
    chosen = logits.gather(1, torch.tensor(completion_ids, device="cuda")[:, None])[:, 0]
    gap = logits.max(dim=1).values - chosen
    worst = gap.argmax().item()
    assert gap[worst].item() <= LOGIT_TOL, f"token {worst} is {gap[worst].item():.3f} logits below the reference argmax"


def generate(llm, prompts):
    from nanovllm import SamplingParams
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MAX_TOKENS), use_tqdm=False)
    return [output["token_ids"] for output in outputs]


def test_greedy_matches_reference(llm, reference):
    prompts = [chat(llm.tokenizer, q) for q in QUESTIONS]
    scheduler = llm.scheduler
    scheduler.reset_stats()
    for prompt, completion in zip(prompts, generate(llm, prompts)):
        assert completion
        check(reference, prompt, completion)
    if scheduler.has_draft_model and not scheduler.has_eagle_head and scheduler.num_draft_tokens and os.path.samefile(DRAFT_MODEL, MODEL):
        # The target drafting for itself: every draft matches the greedy token unless bf16 noise
        # between the draft and verification passes flips a near-tie.
        assert scheduler.num_accepted_tokens / scheduler.num_draft_tokens > 0.8


def test_chunked_prefill_and_prefix_cache_match_reference(llm, reference):
    prompts = [chat(llm.tokenizer, f"{DOCUMENT}\n\n{q}") for q in DOCUMENT_QUESTIONS]
    assert len(prompts[0]) > 1024
    first = generate(llm, prompts)
    # The second call must hit the prefix cache that the first call left behind.
    assert llm.scheduler.block_manager.can_allocate(Sequence(prompts[0])) > 0
    second = generate(llm, prompts)
    for prompt, a, b in zip(prompts, first, second):
        check(reference, prompt, a)
        check(reference, prompt, b)


COPY_TEXT = " ".join(f"Sentence {i}: the scheduler batches requests, the block manager hands out KV cache blocks, "
                     f"and the model runner turns them into input tensors." for i in range(3))


def test_ragged_batch_matches_reference(llm, reference):
    # speculative_per_sequence: a copying task keeps its n-gram drafts (the text is in its prompt)
    # while the questions lose theirs, so the batch decodes with different numbers of drafts per
    # sequence (ragged steps, padded or varlen, and the sampler's ragged branch). With the target
    # drafting for itself, only a near-tie flipped by bf16 noise cuts a sequence. Thinking is off:
    # with it on, the copying task spends its MAX_TOKENS thinking and never starts to copy (A40
    # jobs 233732 and 234187 saw no ragged step at all).
    scheduler = llm.scheduler
    if not scheduler.speculative_per_sequence:
        pytest.skip("per-sequence k only")
    prompts = [chat(llm.tokenizer, f"Repeat the following text exactly, word for word:\n\n{COPY_TEXT}", thinking=False)] + [chat(llm.tokenizer, q, thinking=False) for q in QUESTIONS]
    scheduler.reset_stats()
    for prompt, completion in zip(prompts, generate(llm, prompts)):
        assert completion
        check(reference, prompt, completion)
    if not scheduler.has_draft_model:
        assert scheduler.num_ragged_steps, "the copying task keeps its n-gram drafts while the questions lose theirs"


def test_sampled_drafts_from_the_target_itself_are_nearly_all_accepted(llm):
    # Exact speculative sampling with the target drafting for itself: q is p up to kernel noise
    # (CUDA graph against eager, bf16), so min(1, p/q) accepts nearly every draft at temperature 1.
    # Argmax drafts are accepted with p(argmax), the model's confidence, far below that.
    from nanovllm import SamplingParams
    scheduler = llm.scheduler
    if not scheduler.has_draft_model or scheduler.has_eagle_head or not os.path.samefile(DRAFT_MODEL, MODEL) or scheduler.speculative_min_acceptance is not None or scheduler.speculative_disable_by_batch_size is not None:
        pytest.skip("a draft model drafting for itself with every draft only")
    prompts = [chat(llm.tokenizer, q) for q in QUESTIONS]
    scheduler.reset_stats()
    llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, ignore_eos=True), use_tqdm=False)
    acceptance = scheduler.num_accepted_tokens / scheduler.num_draft_tokens
    if llm.model_runner.sampled_drafts:
        assert acceptance > 0.9, acceptance
    else:
        assert 0.2 < acceptance < 0.95, acceptance


def test_sampling_returns_max_tokens(llm):
    from nanovllm import SamplingParams
    prompts = [chat(llm.tokenizer, q) for q in QUESTIONS]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=MAX_TOKENS, ignore_eos=True), use_tqdm=False)
    assert [len(output["token_ids"]) for output in outputs] == [MAX_TOKENS] * len(prompts)
