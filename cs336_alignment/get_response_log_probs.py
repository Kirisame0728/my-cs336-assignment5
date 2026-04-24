import torch
from transformers import PreTrainedModel
from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    outputs = model(input_ids).logits.float()
    log_probs_all = torch.log_softmax(outputs, dim=-1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        token_entropy = compute_entropy(outputs)
        out = {
            "log_probs": log_probs,
            "token_entropy": token_entropy
        }
        return out

    out = {
        "log_probs": log_probs
    }
    return out