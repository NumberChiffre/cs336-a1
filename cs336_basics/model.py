import torch
import torch.nn as nn
import math
import einops


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        mean = 0
        std = math.sqrt(2 / (in_features + out_features))
        lb, ub = -3 * std, 3 * std
        nn.init.trunc_normal_(weight, mean=mean, std=std, a=lb, b=ub)
        self.w = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(einops.reduce(x ** 2, "... d_model -> ... 1", "mean") + self.eps)
        x = x * self.g / rms
        return x.to(in_dtype)
