from typing import List, Tuple
import torch


class SpeculativeEngine:

  def __init__(self, gamma: int = 4):
    self.gamma = gamma

  def verify_tokens(
      self,
      draft_tokens: torch.Tensor,
      draft_probs: torch.Tensor,
      target_probs: torch.Tensor,
  ) -> Tuple[List[int], int]:
    accepted = []
    seq_len = draft_tokens.shape[1]

    for i in range(seq_len):
      token_id = draft_tokens[0, i].item()
      p_draft = draft_probs[0, i, token_id].item()
      p_target = target_probs[0, i, token_id].item()

      if p_draft <= 0.0:
        accept_prob = 1.0 if p_target > 0 else 0.0
      else:
        accept_prob = min(1.0, p_target / p_draft)

      rand_val = torch.rand(1).item()
      if rand_val <= accept_prob:
        accepted.append(token_id)
      else:
        residual = torch.clamp(target_probs[0, i] - draft_probs[0, i], min=0.0)
        residual_sum = residual.sum()
        if residual_sum > 0:
          residual = residual / residual_sum
          recovery_token = torch.multinomial(residual, num_samples=1).item()
        else:
          recovery_token = torch.argmax(target_probs[0, i]).item()
        accepted.append(recovery_token)
        break

    return accepted, len(accepted)