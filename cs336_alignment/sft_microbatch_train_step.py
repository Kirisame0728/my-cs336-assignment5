import torch
from cs336_alignment.masked_normalize import masked_normalize

def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized_sum = -masked_normalize(policy_log_probs, response_mask, normalize_constant, 1)
    loss = normalized_sum.mean() / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "num_response_tokens": response_mask.sum(),
        "mean_log_prob": normalized_sum.detach(),
    }
    return loss, metadata


