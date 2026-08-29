from paged_infer.memory.block_allocator import BlockAllocator

allocator = BlockAllocator(num_blocks=10, block_size=16)

print("--- [1] Initial Memory State ---")
print(f"Total Free Blocks: {allocator.get_num_free_blocks()} / 10\n")

print("--- [2] Allocating Sequence A (Prompt: 35 tokens) ---")
blocks_a = allocator.allocate_sequence(seq_id="req-user-101", prompt_token_count=35)
print(f"Assigned Physical Blocks: {blocks_a}")
print(f"Remaining Free Blocks: {allocator.get_num_free_blocks()}\n")

print("--- [3] Generating Tokens (Triggering Dynamic Expansion) ---")
for token_idx in range(36, 49):
    allocator.append_slot("req-user-101", current_token_count=token_idx)

new_block = allocator.append_slot("req-user-101", current_token_count=48)
print(f"Boundary reached! New block allocated: Block #{new_block}")
print(f"Updated Block Table: {allocator.block_tables['req-user-101']}\n")

print("--- [4] Releasing Sequence A ---")
allocator.free_sequence("req-user-101")
print(f"Memory Reclaimed. Total Free Blocks: {allocator.get_num_free_blocks()} / 10")