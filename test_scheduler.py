import time
from paged_infer.memory.block_allocator import BlockAllocator
from paged_infer.engine.sequence import Sequence
from paged_infer.engine.scheduler import Scheduler

# 8 physical blocks available (16 tokens each = 128 tokens total capacity)
allocator = BlockAllocator(num_blocks=8, block_size=16)
scheduler = Scheduler(block_allocator=allocator, max_batch_size=3)

# Add two initial requests with different generation lengths
req1 = Sequence(seq_id="Req-1", prompt_tokens=[101, 102, 103], max_new_tokens=3)   # Short request (3 tokens)
req2 = Sequence(seq_id="Req-2", prompt_tokens=[201, 202, 203, 204], max_new_tokens=7) # Medium request (7 tokens)

scheduler.add_sequence(req1)
scheduler.add_sequence(req2)

print("=== Starting Continuous Batching Engine Simulation ===\n")

iteration = 1
while scheduler.waiting or scheduler.running:
    print(f"--- Iteration #{iteration} ---")
    
    # Mid-stream request injection at Iteration #2
    if iteration == 2:
        req3 = Sequence(seq_id="Req-3 (Injected)", prompt_tokens=[301, 302], max_new_tokens=4)
        print(">> [Incoming Traffic] New Request 3 arrived dynamically!")
        scheduler.add_sequence(req3)

    # Schedule step
    current_batch = scheduler.schedule()
    print(f"Active Batch: {[s.seq_id for s in current_batch]} | Free Blocks: {allocator.get_num_free_blocks()}/8")

    # Simulate 1 token generation step for each active request
    for seq in current_batch:
        allocator.append_slot(seq.seq_id, seq.get_len())
        seq.append_token(token_id=999)  # Simulated token ID

    # Clean up completed sequences and reclaim memory
    finished_seqs = scheduler.post_step()
    for f in finished_seqs:
        print(f"✅ Finished: {f.seq_id} (Generated {len(f.output_tokens)} tokens) -> KV Memory Reclaimed!")

    print()
    iteration += 1