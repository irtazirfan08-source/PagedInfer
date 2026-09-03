import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from paged_infer.engine.speculative import SpeculativeEngine


def run_speculative_test():
  engine = SpeculativeEngine(gamma=4)
  vocab_size = 100
  seq_len = 4

  draft_tokens = torch.tensor([[12, 34, 56, 78]])
  draft_probs = torch.zeros(1, seq_len, vocab_size)
  target_probs = torch.zeros(1, seq_len, vocab_size)

  for i, token in enumerate(draft_tokens[0]):
    draft_probs[0, i, token] = 0.75
    target_probs[0, i, token] = 0.85

  accepted, count = engine.verify_tokens(
      draft_tokens, draft_probs, target_probs
  )
  print(f"Verified Tokens: {accepted}")
  print(f"Accepted Count: {count}/{seq_len}")


if __name__ == "__main__":
  run_speculative_test()