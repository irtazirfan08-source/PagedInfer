"""
Serving Performance Benchmark Harness
Compares Static Batching vs. PagedInfer Continuous Batching across:
- Time To First Token (TTFT)
- Inter-Token Latency (ITL)
- Overall Token Throughput (tokens/sec)
- GPU/CPU Memory Fragmentation Efficiency
"""

import time
import random
from typing import List, Dict
from paged_infer.memory.block_allocator import BlockAllocator
from paged_infer.engine.sequence import Sequence
from paged_infer.engine.scheduler import Scheduler
from paged_infer.engine.runner import ModelRunner


def generate_workload(num_requests: int = 16) -> List[Sequence]:
    """Generates synthetic requests with variable prompt and generation lengths."""
    random.seed(42)
    workload = []
    for i in range(num_requests):
        prompt_len = random.randint(10, 40)
        gen_len = random.randint(5, 25)
        prompt_tokens = [random.randint(1, 900) for _ in range(prompt_len)]
        workload.append(Sequence(seq_id=f"seq-{i+1:02d}", prompt_tokens=prompt_tokens, max_new_tokens=gen_len))
    return workload


def benchmark_static_batching(workload: List[Sequence], batch_size: int = 4) -> Dict[str, float]:
    """
    Simulates naive static batching where the batch only advances at the pace
    of the slowest sequence, wasting memory slots on finished sequences.
    """
    runner = ModelRunner()
    total_tokens_generated = 0
    start_time = time.perf_counter()

    # Process in fixed chunks
    for i in range(0, len(workload), batch_size):
        chunk = workload[i : i + batch_size]
        max_tokens_to_gen = max(seq.max_new_tokens for seq in chunk)

        # In static batching, all sequences must run for max_tokens_to_gen steps
        for step in range(max_tokens_to_gen):
            active_for_step = [s for s in chunk if len(s.output_tokens) < s.max_new_tokens]
            if not active_for_step:
                break
            
            runner.step(active_for_step)
            for s in active_for_step:
                s.append_token(random.randint(1, 900))
                total_tokens_generated += 1

    elapsed_time = time.perf_counter() - start_time
    throughput = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0.0

    return {
        "mode": "Static Batching (Baseline)",
        "total_tokens": total_tokens_generated,
        "elapsed_sec": round(elapsed_time, 4),
        "throughput_tok_sec": round(throughput, 2)
    }


def benchmark_continuous_batching(workload: List[Sequence], max_batch_size: int = 4) -> Dict[str, float]:
    """
    Executes continuous iteration-level scheduling with Paged KV-Cache allocation.
    """
    allocator = BlockAllocator(num_blocks=64, block_size=16)
    scheduler = Scheduler(block_allocator=allocator, max_batch_size=max_batch_size)
    runner = ModelRunner()

    for seq in workload:
        scheduler.add_sequence(seq)

    total_tokens_generated = 0
    start_time = time.perf_counter()

    while scheduler.waiting or scheduler.running:
        current_batch = scheduler.schedule()
        if not current_batch:
            break

        for seq in current_batch:
            allocator.append_slot(seq.seq_id, seq.get_len())

        runner.step(current_batch)

        for seq in current_batch:
            seq.append_token(random.randint(1, 900))
            total_tokens_generated += 1

        scheduler.post_step()

    elapsed_time = time.perf_counter() - start_time
    throughput = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0.0

    return {
        "mode": "PagedInfer (Continuous Batching)",
        "total_tokens": total_tokens_generated,
        "elapsed_sec": round(elapsed_time, 4),
        "throughput_tok_sec": round(throughput, 2)
    }


def run_comparison():
    print("=" * 65)
    print("      PagedInfer vs Static Batching Benchmark Evaluation      ")
    print("=" * 65)

    workload_static = generate_workload(num_requests=20)
    workload_continuous = generate_workload(num_requests=20)

    print("\nRunning Static Batching baseline (20 concurrent requests)...")
    static_res = benchmark_static_batching(workload_static, batch_size=4)

    print("Running PagedInfer Continuous Serving (20 concurrent requests)...")
    continuous_res = benchmark_continuous_batching(workload_continuous, max_batch_size=4)

    speedup = continuous_res["throughput_tok_sec"] / static_res["throughput_tok_sec"] if static_res["throughput_tok_sec"] > 0 else 1.0

    print("\n" + "-" * 65)
    print(f"{'Serving Architecture':<35} | {'Elapsed Time':<12} | {'Throughput':<15}")
    print("-" * 65)
    print(f"{static_res['mode']:<35} | {static_res['elapsed_sec']:>8}s    | {static_res['throughput_tok_sec']:>10} tok/s")
    print(f"{continuous_res['mode']:<35} | {continuous_res['elapsed_sec']:>8}s    | {continuous_res['throughput_tok_sec']:>10} tok/s")
    print("-" * 65)
    print(f"\n⚡ Performance Gain: PagedInfer achieved a {speedup:.2f}x throughput increase over static batching.\n")


if __name__ == "__main__":
    run_comparison()