from typing import List, Tuple
from paged_infer.memory.block_allocator import BlockAllocator
from paged_infer.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    Continuous iteration-level scheduler.
    Dynamically schedules prefill (new prompts) and decode (token steps)
    while managing physical memory via the BlockAllocator.
    """
    def __init__(self, block_allocator: BlockAllocator, max_batch_size: int = 4):
        self.block_allocator = block_allocator
        self.max_batch_size = max_batch_size
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    def add_sequence(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], List[Sequence]]:
        """
        Runs before each forward pass:
        1. Retains active running sequences.
        2. Ingests waiting sequences if memory and batch size permit.
        """
        # 1. Admit new requests from waiting queue if memory allows
        while self.waiting and len(self.running) < self.max_batch_size:
            seq = self.waiting[0]
            num_blocks = (len(seq.prompt_tokens) + self.block_allocator.block_size - 1) // self.block_allocator.block_size

            if self.block_allocator.can_allocate(num_blocks):
                self.waiting.pop(0)
                self.block_allocator.allocate_sequence(seq.seq_id, len(seq.prompt_tokens))
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                # Not enough memory blocks currently free
                break

        # Return running sequences for the current forward step
        return self.running

    def post_step(self) -> List[Sequence]:
        """
        Runs after each token generation step:
        Reclaims memory for finished requests and removes them from the running batch.
        """
        finished: List[Sequence] = []
        active: List[Sequence] = []

        for seq in self.running:
            if seq.is_finished():
                self.block_allocator.free_sequence(seq.seq_id)
                finished.append(seq)
            else:
                active.append(seq)

        self.running = active
        return finished