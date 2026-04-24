import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    probs = torch.exp(log_probs)
    entropy = torch.sum(- probs * log_probs, dim=-1)
    return entropy
