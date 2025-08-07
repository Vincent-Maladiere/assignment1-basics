import torch
from torch import nn
import einops


def _init_W(in_features, out_features, sigma_sq, device, dtype):
    t = torch.empty((in_features, out_features), device=device, dtype=dtype)
    sigma = sigma_sq**0.5
    torch.nn.init.trunc_normal_(t, std=sigma_sq, a=-3 * sigma, b=3 * sigma)
    return nn.Parameter(t, requires_grad=True)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = _init_W(
            in_features,
            out_features,
            sigma_sq=2 / (in_features + out_features),
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return einops.einsum(x, self.W, "... k, k l -> ... l")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.W = _init_W(
            num_embeddings,
            embedding_dim,
            sigma_sq=1,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids):
        return self.W[token_ids]
