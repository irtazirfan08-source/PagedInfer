from paged_infer.memory.block_allocator import BlockAllocator


def test_prefix_caching_hit_and_sharing():
    allocator = BlockAllocator(num_blocks=8, block_size=16)

    # 32 tokens = exactly 2 physical blocks (16 tokens each)
    shared_prefix = list(range(100, 132))

    # Request A: Cold start prompt
    table_a = allocator.allocate("seq_A", num_tokens=32, prompt_tokens=shared_prefix)
    assert len(table_a) == 2, f"Expected 2 blocks, got {len(table_a)}"
    assert allocator.get_num_free_blocks() == 6
    assert allocator.cache_hits == 0

    # Request B: Same prompt prefix + extra novel tokens (34 total -> 3 blocks)
    prompt_b = shared_prefix + [999, 998]
    table_b = allocator.allocate("seq_B", num_tokens=34, prompt_tokens=prompt_b)

    # Assert block sharing and reference counting
    assert table_b[0] == table_a[0], "First physical block was not shared"
    assert table_b[1] == table_a[1], "Second physical block was not shared"
    assert table_b[2] not in table_a, "Novel block should not belong to seq_A"
    assert allocator.cache_hits == 2, f"Expected 2 cache hits, got {allocator.cache_hits}"
    assert allocator.get_num_free_blocks() == 5, f"Expected 5 free blocks, got {allocator.get_num_free_blocks()}"
    print("--- [1] Prefix block sharing and cache hits verified ---")
    print(f"Seq A Table: {table_a} | Seq B Table: {table_b}")
    print(f"Cache Hits: {allocator.cache_hits} | Free Blocks: {allocator.get_num_free_blocks()}/8")


def test_ref_count_lifecycle():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    prefix = list(range(16))  # 1 full block

    allocator.allocate("seq_1", num_tokens=16, prompt_tokens=prefix)
    allocator.allocate("seq_2", num_tokens=16, prompt_tokens=prefix)

    # Free seq_1; shared physical block must persist for seq_2
    allocator.free("seq_1")
    assert allocator.get_num_free_blocks() == 3, "Block freed prematurely while still held by seq_2"
    assert allocator.block_ref_counts[0] == 1, "Ref count should decrement to 1"

    # Free seq_2; physical block should now be returned to the pool
    allocator.free("seq_2")
    assert allocator.get_num_free_blocks() == 4, "Physical block was not returned to free list"
    assert allocator.block_ref_counts[0] == 0, "Ref count should be 0"
    print("\n--- [2] Reference counting reclamation lifecycle verified ---")
    print("Zero-leakage memory reclamation confirmed.")


if __name__ == "__main__":
    print("=== Running PagedInfer Prefix Caching Test Suite ===")
    test_prefix_caching_hit_and_sharing()
    test_ref_count_lifecycle()
    print("\nAll prefix cache tests passed successfully.")