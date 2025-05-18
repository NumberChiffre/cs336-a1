import torch
import torch.nn as nn
import math
from einops import einsum, rearrange, reduce


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # [batch, seq_len, d_k] -> [batch, seq_len, 1]
    x_max = x.max(dim=dim, keepdim=True).values
    # [batch, seq_len, d_k] -> [batch, seq_len, d_k]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]
    # [batch, seq_len, d_k] @ [batch, seq_len, d_k] -> [batch, seq_len, seq_len]
    q_k = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    scaled_q_k = q_k / math.sqrt(d_k)
    # [batch, seq_len, seq_len] -> [batch, seq_len, seq_len]
    scaled_q_k = torch.where(mask, scaled_q_k, float("-inf"))
    # [batch, seq_len, seq_len] -> [batch, seq_len, d_v]
    attn = einsum(softmax(scaled_q_k, dim=-1), V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")
    return attn


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
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, d_in] @ [d_out, d_in] -> [batch, seq_len, d_out]
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        weight = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        mean = 0
        std = 1
        lb, ub = -3, 3
        nn.init.trunc_normal_(weight, mean=mean, std=std, a=lb, b=ub)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # [batch, seq_len, d_model] -> [batch, seq_len, 1]
        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        x = x * self.weight / rms
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = (8 * d_model) // 3
        assert d_ff % 64 == 0, "`d_ff` must be a multiple of 64"
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, d_ff]
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        # [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        swiglu = self.w2(silu * self.w3(x))
        return swiglu


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        # [d_k / 2]
        kd = torch.arange(0, d_k, 2, device=device) / d_k
        # [max_seq_len, 1]
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        # [max_seq_len, d_k / 2]
        angles = positions / (theta**kd)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, d_k]
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        # [batch, seq_len, d_k // 2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos_pos - x_odd * sin_pos
        rot_odd = x_even * sin_pos + x_odd * cos_pos
        # [batch, seq_len, d_k, 2] with last axis: 0 = even, 1 = odd
        rot = rearrange([rot_even, rot_odd], "pair ... -> ... pair")
        # [batch, seq_len, d_k * 2]
        rot = rearrange(rot, "... d_k pair -> ... (d_k pair)")
        return rot


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.qkv_proj = Linear(in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype)
        self.o_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [batch, seq_len, d_model] -> [batch, seq_len, 3 * d_model]
        seq_len = x.size(1)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=2)
        # [batch, seq_len, 3 * d_model] -> [batch, seq_len, num_heads, d_k]
        q = rearrange(q, "batch seq_len (h d_k) -> batch h seq_len d_k", h=self.num_heads)
        k = rearrange(k, "batch seq_len (h d_k) -> batch h seq_len d_k", h=self.num_heads)
        v = rearrange(v, "batch seq_len (h d_k) -> batch h seq_len d_k", h=self.num_heads)
        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool))
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "batch h seq_len d_k -> batch seq_len (h d_k)")
        return self.o_proj(attn)
