import torch


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask.to(tensor.dtype)
    sum = torch.sum(masked_tensor, dim=dim)
    return sum / normalize_constant