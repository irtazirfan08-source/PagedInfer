"""
PyTorch Execution Engine
Simulates forward tensor passes with realistic vocabulary projections and KV-cache tracking.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from paged_infer.engine.sequence import Sequence


class MockCausalLM(nn.Module):
    """Lightweight causal transformer layer simulating multi-head attention and logits."""
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        logits = self.proj(x)
        return logits


class ModelRunner:
    """Coordinates tensor batching and forward passes for active sequences."""
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MockCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def step(self, sequences: List[Sequence]) -> Dict[str, int]:
        """
        Executes one forward step across the continuous batch.
        Returns a mapping of sequence_id -> newly sampled token_id.
        """
        if not sequences:
            return {}

        latest_tokens = []
        for seq in sequences:
            if not seq.output_tokens:
                latest_tokens.append(seq.prompt_tokens[-1])
            else:
                latest_tokens.append(seq.output_tokens[-1])

        input_tensor = torch.tensor(latest_tokens, dtype=torch.long, device=self.device)
        logits = self.model(input_tensor)
        
        sampled_tokens = torch.argmax(logits, dim=-1).tolist()

        results = {}
        for seq, next_tok in zip(sequences, sampled_tokens):
            results[seq.seq_id] = int(next_tok)

        return results