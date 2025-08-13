import math
from functools import wraps
import torch
from torch import nn
import einops
from collections import OrderedDict


def _linear_sigma_sq(d_in, d_out):
    return 2 / (d_in + d_out)


def _init_W(in_features, out_features, device=None, dtype=None, use_linear_sigma=True):
    t = torch.empty((out_features, in_features), device=device, dtype=dtype)
    if use_linear_sigma:
        sigma_sq = _linear_sigma_sq(in_features, out_features)
    else:
        sigma_sq = 1
    sigma = sigma_sq**0.5
    torch.nn.init.trunc_normal_(t, std=sigma_sq, a=-3 * sigma, b=3 * sigma)
    return nn.Parameter(t, requires_grad=True)


def _dot_product(*tensors):
    return einops.einsum(*tensors, "... k, l k -> ... l")


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = _init_W(
            in_features,
            out_features,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return _dot_product(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = _init_W(
            embedding_dim,
            num_embeddings,
            device=device,
            dtype=dtype,
            use_linear_sigma=False,
        )

    def forward(self, token_ids):
        return self.weight[token_ids]


def _init_rms_gain(d_model, device):
    t = torch.ones(d_model, dtype=torch.float32, device=device)
    return nn.Parameter(t, requires_grad=True)


def cast_precision(target_dtype):
    def decorator(func):
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            x = x.to(target_dtype)
            res = func(self, x, *args, **kwargs)
            return res

        return wrapper

    return decorator


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = _init_rms_gain(d_model, device)

    @cast_precision(torch.float32)
    def forward(self, x):
        z = x**2 + self.eps
        rms = einops.reduce(z, "... k -> ... 1", "mean") ** 0.5
        return (x / rms) * self.weight


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        y = self.w1(x)
        z = self.w3(x)
        return self.w2(silu(y) * z)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self._init_rot_matrix()

    def forward(self, x, token_positions):
        x = einops.rearrange(x, "... s (k n) -> ... s k n", n=2)
        # FIXME: self.R[token_positions] messed with batch dimensions during training.
        if self.training:
            R = self.R
        else:
            R = self.R[token_positions]
        x = einops.einsum(x, R, "... s k n, ... s k m n -> ... s k m")
        return einops.rearrange(x, "... s k n -> ... s (k n)", n=2)

    def _init_rot_matrix(self):
        t = torch.empty(self.max_seq_len, self.d_k // 2, 2, 2, device=self.device)
        for i in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                freqs = self.theta ** (-2 * k / self.d_k)
                theta_ik = i * freqs
                t[i][k] = torch.tensor(
                    [
                        [math.cos(theta_ik), -math.sin(theta_ik)],
                        [math.sin(theta_ik), math.cos(theta_ik)],
                    ]
                )
        self.register_buffer("R", t, persistent=False)


def softmax(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    exps = (t - t.max(dim=dim, keepdim=True).values).exp()
    return exps / exps.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    A = einops.einsum(Q, K, "... s_q d, ... s_k d -> ... s_q s_k")
    A /= d_k**0.5

    if mask is not None:
        A = A + torch.where(
            mask,
            torch.tensor(0.0, device=A.device, dtype=A.dtype),
            torch.tensor(-torch.inf, device=A.device, dtype=A.dtype),
        )
    A = softmax(A, dim=-1)

    return einops.einsum(A, V, "... s_q s_k, ... s_k d -> ... s_q d")


class MHA(nn.Module):
    def __init__(self, d_model, n_heads, rope=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope = rope
        self.device = device
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.output_proj = Linear(d_model, d_model, device=device)

    def forward(self, x, token_positions=None):
        # Combine weights W into a single tensor to perform a single matmul.
        Q, K, V = einops.rearrange(
            [self.q_proj(x), self.k_proj(x), self.v_proj(x)],
            "a ... s (n d) -> a ... n s d",
            a=3,
            n=self.n_heads,
        )

        # Create the causal mask
        seq_len = x.shape[-2]
        device = x.device
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()

        # Run Rope on each heads independently.
        if self.rope is not None:
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        # Run attention in a embarassingly parallel loop.
        attn_out = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal)

        # Concatenate the attention heads outputs and perform the final matmul.
        attn_out = einops.rearrange(attn_out, "... n s d -> ... s (n d)")
        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.rope = RotaryPositionalEmbedding(
            theta, d_model // num_heads, max_seq_len, device=device
        )
        self.ln1 = RMSNorm(d_model, device=device)
        self.attn = MHA(d_model, num_heads, rope=self.rope, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x):
        seq_len = x.shape[-2]
        batch_size = x.shape[-3]
        token_positions = einops.repeat(
            torch.arange(seq_len, device=self.device), "s -> b s", b=batch_size
        )
        x = x + self.attn(
            self.ln1(x),
            token_positions,
        )
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        str(l),
                        TransformerBlock(
                            d_model=d_model,
                            num_heads=num_heads,
                            d_ff=d_ff,
                            max_seq_len=context_length,
                            theta=rope_theta,
                            device=device,
                        ),
                    )
                    for l in range(num_layers)
                ]
            )
        )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(self, x):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        return self.lm_head(x)
