from enum import Enum
from typing import List, Optional


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


class Sequence:
    """Represents a single generation request and its token lifecycle."""
    def __init__(self, seq_id: str, prompt_tokens: List[int], max_new_tokens: int = 20):
        self.seq_id: str = seq_id
        self.prompt_tokens: List[int] = prompt_tokens
        self.output_tokens: List[int] = []
        self.max_new_tokens: int = max_new_tokens
        self.status: SequenceStatus = SequenceStatus.WAITING

    def get_len(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    def is_finished(self) -> bool:
        return len(self.output_tokens) >= self.max_new_tokens

    def append_token(self, token_id: int) -> None:
        self.output_tokens.append(token_id)
        if self.is_finished():
            self.status = SequenceStatus.FINISHED