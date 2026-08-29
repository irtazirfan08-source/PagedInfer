"""
Paged KV-Cache Block Allocator
Manages physical GPU/CPU memory blocks for token generation,
preventing memory fragmentation by allocating fixed-size pages.
"""

from typing import List, Dict, Optional


class PhysicalBlock:
    """Represents one physical memory slot capable of holding `block_size` tokens."""
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id: int = block_id
        self.block_size: int = block_size
        self.ref_count: int = 0

    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockAllocator:
    """
    Tracks free and allocated physical blocks.
    Maps a sequence's logical token position to physical memory pages.
    """
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        
        self.free_blocks: List[int] = list(range(num_blocks))
        self.all_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        }
        self.block_tables: Dict[str, List[int]] = {}

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_required_blocks: int) -> bool:
        return len(self.free_blocks) >= num_required_blocks

    def allocate_sequence(self, seq_id: str, prompt_token_count: int) -> List[int]:
        num_blocks_needed = (prompt_token_count + self.block_size - 1) // self.block_size

        if not self.can_allocate(num_blocks_needed):
            raise MemoryError(
                f"Out of Memory: Needed {num_blocks_needed} blocks, but only {len(self.free_blocks)} available."
            )

        allocated_ids: List[int] = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            self.all_blocks[block_id].ref_count = 1
            allocated_ids.append(block_id)

        self.block_tables[seq_id] = allocated_ids
        return allocated_ids

    def append_slot(self, seq_id: str, current_token_count: int) -> Optional[int]:
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not registered in block table.")

        if current_token_count % self.block_size == 0:
            if len(self.free_blocks) == 0:
                raise MemoryError("KV-Cache Exhausted during token decoding phase.")
            
            new_block_id = self.free_blocks.pop(0)
            self.all_blocks[new_block_id].ref_count = 1
            self.block_tables[seq_id].append(new_block_id)
            return new_block_id
        
        return self.block_tables[seq_id][-1]

    def free_sequence(self, seq_id: str) -> None:
        if seq_id not in self.block_tables:
            return

        for block_id in self.block_tables[seq_id]:
            block = self.all_blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block_id)

        del self.block_tables[seq_id]