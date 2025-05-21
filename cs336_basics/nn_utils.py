from collections.abc import Iterable
import torch


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_max = logits.max(dim=-1, keepdim=True).values
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    sum_exp = (logits - logits_max).exp().sum(dim=-1, keepdim=True)
    log_sum_exp = sum_exp.log()
    return -((target_logits - logits_max) - log_sum_exp).mean()


def gradient_clip(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in params if p.grad is not None]
    squared_norms = [torch.sum(grad ** 2) for grad in grads]
    l2_norm = torch.sqrt(sum(squared_norms) + eps)
    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for grad in grads:
            grad.detach().mul_(scale)
