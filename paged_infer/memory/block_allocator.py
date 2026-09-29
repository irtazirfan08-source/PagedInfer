import math
from typing import Dict, List, Optional, Tuple


class BlockAllocator:
    """Manages physical KV-cache memory blocks with dynamic allocation,
    reference-counted prefix caching, and zero-fragmentation reclamation.
    """

    def __init__(self, num_blocks: int = 32, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[str, List[int]] = {}

        # Prefix caching state
        self.prefix_cache: Dict[Tuple[int, ...], int] = {}
        self.block_ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}
        self.cache_hits: int = 0

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        required_blocks = math.ceil(num_tokens / self.block_size)
        return len(self.free_blocks) >= required_blocks

    def get_block_table(self, seq_id: str) -> List[int]:
        return self.block_tables.get(seq_id, [])

    def allocate(
        self,
        seq_id: str,
        num_tokens: int,
        prompt_tokens: Optional[List[int]] = None,
    ) -> List[int]:
        """Allocates physical blocks for a sequence. Reuses cached blocks for
        full token chunks of size `block_size` when prompt_tokens are supplied.
        """
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already has allocated blocks.")

        num_required_blocks = math.ceil(num_tokens / self.block_size)
        allocated_table: List[int] = []

        if prompt_tokens:
            num_full_chunks = len(prompt_tokens) // self.block_size
            for chunk_idx in range(num_full_chunks):
                chunk = tuple(
                    prompt_tokens[chunk_idx * self.block_size : (chunk_idx + 1) * self.block_size]
                )
                if chunk in self.prefix_cache:
                    cached_block = self.prefix_cache[chunk]
                    self.block_ref_counts[cached_block] += 1
                    allocated_table.append(cached_block)
                    self.cache_hits += 1
                else:
                    if not self.free_blocks:
                        self._rollback_partial_allocation(allocated_table)
                        raise MemoryError("Out of physical memory blocks during prefix allocation.")
                    new_block = self.free_blocks.pop(0)
                    self.prefix_cache[chunk] = new_block
                    self.block_ref_counts[new_block] = 1
                    allocated_table.append(new_block)

        remaining_blocks_needed = num_required_blocks - len(allocated_table)
        if remaining_blocks_needed > len(self.free_blocks):
            self._rollback_partial_allocation(allocated_table)
            raise MemoryError("Out of physical memory blocks.")

        for _ in range(remaining_blocks_needed):
            block_id = self.free_blocks.pop(0)
            self.block_ref_counts[block_id] = 1
            allocated_table.append(block_id)

        self.block_tables[seq_id] = allocated_table
        return allocated_table

    def allocate_sequence(self, seq_id: str, prompt_token_count: int) -> List[int]:
        """Backward-compatible entry point for baseline memory allocation."""
        return self.allocate(seq_id=seq_id, num_tokens=prompt_token_count)

    def append_slot(
        self,
        seq_id: str,
        current_token_count: Optional[int] = None,
        current_len: Optional[int] = None,
        **kwargs,
    ) -> Optional[int]:
        """Allocates a new block if sequence length crosses a block boundary."""
        token_count = current_token_count if current_token_count is not None else current_len
        if token_count is None:
            raise ValueError("Token count must be provided to append_slot.")

        if token_count % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError(f"Out of physical memory blocks for sequence {seq_id}.")
            new_block = self.free_blocks.pop(0)
            self.block_ref_counts[new_block] = 1
            self.block_tables[seq_id].append(new_block)
            return new_block
        return None

    def free(self, seq_id: str) -> None:
        """Decrements reference counts and reclaims blocks whose count drops to zero."""
        if seq_id not in self.block_tables:
            return

        blocks = self.block_tables.pop(seq_id)
        for block_id in blocks:
            self.block_ref_counts[block_id] -= 1
            if self.block_ref_counts[block_id] == 0:
                dead_keys = [k for k, v in self.prefix_cache.items() if v == block_id]
                for k in dead_keys:
                    del self.prefix_cache[k]
                self.free_blocks.append(block_id)

        self.free_blocks.sort()

    def free_sequence(self, seq_id: str) -> None:
        """Backward-compatible entry point for freeing memory."""
        self.free(seq_id)

    def _rollback_partial_allocation(self, partial_blocks: List[int]) -> None:
        for block_id in partial_blocks:
            self.block_ref_counts[block_id] -= 1
            if self.block_ref_counts[block_id] == 0:
                dead_keys = [k for k, v in self.prefix_cache.items() if v == block_id]
                for k in dead_keys:
                    del self.prefix_cache[k]
                self.free_blocks.append(block_id)
        self.free_blocks.sort()