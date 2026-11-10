"""Speculative decoding benchmark: throughput and draft acceptance without drafts, with n-gram
drafts, and with a draft model (--speculative-model).

The "edit" workload asks the model to return a source file with docstrings added, so most of the
output repeats the prompt (where prompt lookup helps); the "chat" workload asks open questions.
"""
import argparse
import glob
import json
import os
import time

from nanovllm import LLM, SamplingParams

QUESTIONS = [
    "Explain how a KV cache speeds up transformer inference.",
    "Write a short story about a robot learning to paint.",
    "What are the trade-offs between tensor parallelism and pipeline parallelism?",
    "Describe the water cycle to a ten-year-old.",
    "List ten tips for writing clean Python code, with an example for each.",
    "Compare TCP and UDP.",
    "How does a hash map handle collisions?",
    "Summarize the history of the printing press.",
]


def build_prompts(tokenizer, workload, num_prompts, max_prompt_tokens):
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanovllm", "**", "*.py"), recursive=True))
    edits = []
    for path in files:
        source = open(path).read().strip()
        if source and len(tokenizer.encode(source)) <= max_prompt_tokens:
            edits.append(f"Add a one-line docstring to every function and class in this file. Reply with the complete file and nothing else.\n\n```python\n{source}\n```")
    if workload == "edit":
        texts = edits
    elif workload == "chat":
        texts = QUESTIONS
    else:    # mixed: edit and chat prompts alternate, so every batch holds both
        texts = [text for pair in zip(edits, QUESTIONS) for text in pair]
    texts = [texts[i % len(texts)] for i in range(num_prompts)]
    return [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True, enable_thinking=False) for t in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--workload", choices=["edit", "chat", "mixed"], default="edit", help="mixed alternates edit and chat prompts")
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--speculative-model", default=None, help="draft model directory, or an EAGLE-3 head's (speculators format); n-gram drafts if omitted")
    parser.add_argument("--batch-size", type=int, default=1, help="requests per generate call")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--greedy-drafts", action="store_true", help="a draft model proposes its argmax (one-hot q) instead of sampling at the sequence's temperature with min(1, p/q) verification")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--speculative-disable-by-batch-size", type=int, default=None, help="decode without drafts while more sequences than this are running")
    parser.add_argument("--speculative-min-acceptance", type=float, default=None, help="propose only the leading draft positions whose measured acceptance rate reaches this (down to none)")
    parser.add_argument("--probe-interval", type=int, default=None, help="with --speculative-min-acceptance: decode steps between two full-K probes (default Scheduler.PROBE_INTERVAL)")
    parser.add_argument("--speculative-per-sequence", action="store_true", help="with --speculative-min-acceptance: a number of drafts per sequence, from its own acceptance rates")
    parser.add_argument("--speculative-ragged", choices=["varlen", "pad"], default="pad", help="how a batch with different numbers of drafts runs: varlen attention without CUDA graphs, or padded to the most drafts with the fixed-width kernel and the graphs")
    parser.add_argument("--profile", type=int, default=0, metavar="ROWS", help="after the warmup, run one batch under torch.profiler and print its top ROWS ops by CUDA time and by CPU time")
    parser.add_argument("--save-outputs", default=None, metavar="FILE", help="write every request's token ids to FILE (JSON), to compare runs at T=0")
    args = parser.parse_args()

    llm = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=4096, async_scheduling=args.async_scheduling, speculative_disable_by_batch_size=args.speculative_disable_by_batch_size, num_speculative_tokens=args.num_speculative_tokens,
              speculative_model=args.speculative_model, speculative_min_acceptance=args.speculative_min_acceptance, speculative_per_sequence=args.speculative_per_sequence, speculative_ragged=args.speculative_ragged,
              speculative_sampled_drafts=not args.greedy_drafts)
    prompts = build_prompts(llm.tokenizer, args.workload, args.num_prompts, args.max_prompt_tokens)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    llm.generate(prompts[:1], SamplingParams(max_tokens=16), use_tqdm=False)    # warmup

    scheduler = llm.scheduler
    if args.probe_interval is not None:
        scheduler.PROBE_INTERVAL = args.probe_interval
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            llm.generate(prompts[:args.batch_size], sampling_params, use_tqdm=False)
        averages = prof.key_averages()
        print(averages.table(sort_by="cuda_time_total", row_limit=args.profile))
        print(averages.table(sort_by="self_cpu_time_total", row_limit=args.profile))
    scheduler.reset_stats()
    num_tokens = 0
    token_ids = []
    t = time.perf_counter()
    for i in range(0, len(prompts), args.batch_size):
        outputs = llm.generate(prompts[i:i + args.batch_size], sampling_params, use_tqdm=False)
        num_tokens += sum(len(output["token_ids"]) for output in outputs)
        token_ids += [output["token_ids"] for output in outputs]
    t = time.perf_counter() - t
    if args.save_outputs:
        with open(args.save_outputs, "w") as f:
            json.dump(token_ids, f)
    acceptance = scheduler.num_accepted_tokens / scheduler.num_draft_tokens if scheduler.num_draft_tokens else 0.0
    drafter = "none" if not args.num_speculative_tokens else "eagle3" if scheduler.has_eagle_head else "model" if args.speculative_model else "ngram"
    # Per draft position: accepted / proposed (position j is only accepted when the j-1 before it were).
    # k_steps: decode steps by their number of drafts (adaptive speculation picks it per step); with
    # --speculative-per-sequence, sequence-steps. ragged: the share of decode steps whose sequences
    # had different numbers of drafts (run eagerly); capped: the share decoded without drafts by the
    # batch-size rule.
    acc_pos = ",".join(f"{a / p:.2f}" if p else "-" for p, a in zip(scheduler.num_proposed_by_position[1:], scheduler.num_accepted_by_position[1:]))
    k_steps = ",".join(map(str, scheduler.num_steps_by_k))
    steps = max(scheduler.num_decode_steps, 1)
    print(f"workload={args.workload} drafter={drafter} k={args.num_speculative_tokens} async={int(args.async_scheduling)} disable_by={args.speculative_disable_by_batch_size} min_acc={args.speculative_min_acceptance} probe={scheduler.PROBE_INTERVAL} per_seq={int(args.speculative_per_sequence)} ragged_mode={args.speculative_ragged} batch={args.batch_size} T={args.temperature} sampled_drafts={int(not args.greedy_drafts)} "
          f"tokens={num_tokens} time={t:.2f}s throughput={num_tokens / t:.1f}tok/s acceptance={acceptance:.3f} acc_pos={acc_pos} k_steps={k_steps} "
          f"ragged={scheduler.num_ragged_steps / steps:.2f} capped={scheduler.num_capped_steps / steps:.2f}")


if __name__ == "__main__":
    main()
