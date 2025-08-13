import itertools
import math
import torch
import einops
import numpy as np


def cross_entropy(inputs, targets):
    seq_len = inputs.shape[-2]
    vocab_size = inputs.shape[-1]

    logits = inputs - inputs.max(dim=-1, keepdim=True).values
    left = logits.exp().sum(axis=-1).log().ravel()

    logits = einops.rearrange(logits, "... s v -> (... s) v", s=seq_len, v=vocab_size)
    targets = einops.rearrange(targets, "... s -> (... s)", s=seq_len)
    all_seqs = np.prod(inputs.shape[:-1])
    indices = torch.arange(all_seqs, device=inputs.device)
    right = logits[indices, targets]

    scores = left - right

    return einops.reduce(scores, "... -> 1", "mean")


def learning_rate_schedule(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return t / T_w * lr_max
    elif T_w <= t <= T_c:
        return lr_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (
            lr_max - lr_min
        )
    return lr_min


def gradient_clipping(parameters, max_l2_norm):
    l2 = (
        torch.cat([p.grad.data**2 for p in parameters if p.grad is not None]).sum()
        ** 0.5
    )
    if l2 <= max_l2_norm:
        return
    eps = 1e-6
    clipping = max_l2_norm / (l2 + eps)
    for p in parameters:
        if p.grad is None:
            continue
        p.grad *= clipping


class AdamW(torch.optim.Optimizer):
    """
    Adam maths
    ----------
    w_t = w_t-1 - lr * (m_t / sqrt(s_t + eps))
    where:
    m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
    s_t = beta_2 * s_t-1 + (1 - beta_2) * g_t**2

    What about the weight decay uncoupling from the gradient?
    Instead of having it like g_t(theta / t), we have

    w_t = w_t-1 - lr * (m_t / sqrt(s_t + eps)) - lambda * w

    We also need to adjust betas slightly:
    beta_1_t = beta_1 / (1 + t)
    beta_2_t = beta_2 / (1 + t)

    Memory and FLOP accounting
    --------------------------
    a) Peak memory? Assuming float32, which is 4 bytes, for P parameters.

    The optimizer stores 3 elements:
    - Model weights p = 4 P bytes
    - Gradients g = 4 P bytes
    - Variance s = 4 P bytes
    => 12 P bytes

    + Model weight and Gradient outside of the Optimizer state, float16 (2 bytes)
    => 4 P bytes

    Total: 16 P bytes

    Activations memory footprint:

    Act = batch_size * (
    n_layer * (context_length * d_model * 34 + 5 * context_length ** 2)
    + context_length * V
    )

    b) For GPT2-XL, P = num_layer * (4 * d_model ** 2 + 3 * d_model * d_ff + 2 * V * d_model)
    => P = 2B

    Non activation footprint: 2B * 16 = 32 GB
    Remaining: 80GB - 32GB = 48GB

    Act = batch_size * 3GB
    => batch_size = 16

    c) FLOPs for a single step:

    6PD = 2PD (forward) + 4PD (backward)
    where D is the batch size

    C = 6 * 2B * 16 = 96 GFlops / step

    4) How long to train GPT-2XL for 400k steps?
    With MFU = observed throughput / theoretical throughput,
    and MFU = 50%, theoretical throughput = 19.5 TFLOPs (19e12)
    => observed throughput = 10 TFLOps = 1e13 FLOP/s

    Batch size = 1024
    C = 6 P D = 6 * 2B * 1024 * 400k = 4.9152e+18 FLOPs = 492 PFLOPs

    T = 5e18 / 1e13 = 5e5 s = 5.8 days
    """

    def __init__(
        self,
        params,
        max_l2_norm=None,
        **kwargs,
    ):
        defaults = kwargs
        self.max_l2_norm = max_l2_norm
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        # TODEBUG: some weights set to torch.nan when using gradient clipping.
        # if self.max_l2_norm is not None:
        #     gradient_clipping(
        #         parameters=list(itertools.chain(*self.param_groups)),
        #         max_l2_norm=self.max_l2_norm,
        #     )
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]  # Get the learning rate.
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]

            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 1)
                m_t_prev = state.get("m_t_prev", torch.zeros_like(p.data))
                s_t_prev = state.get("s_t_prev", torch.zeros_like(p.data))
                s_t_prev = s_t_prev.clip(min=0)
                # Get the gradient of loss with respect to p.
                grad = p.grad
                m_t = beta_1 * m_t_prev + (1 - beta_1) * grad
                s_t = beta_2 * s_t_prev + (1 - beta_2) * (grad**2)

                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                # Update weight tensor in-place.
                p.data -= lr_t * (m_t / torch.sqrt(s_t + eps))
                p.data -= lr * weight_decay * p.data

                # Increment iteration number.
                state["t"] = t + 1
                state["m_t_prev"] = m_t
                state["s_t_prev"] = s_t

        return loss


def get_batch(dataset, batch_size, context_length, device):

    x = torch.empty(batch_size, context_length, device=device, dtype=torch.int32)
    y = torch.empty(batch_size, context_length, device=device, dtype=torch.int32)

    max_starting_index = len(dataset) - context_length

    for i, starting_index in enumerate(
        np.random.randint(0, max_starting_index, size=batch_size)
    ):
        x[i] = torch.arange(
            starting_index,
            starting_index + context_length,
            device=device,
            dtype=torch.int32,
        )
        y[i] = x[i] + 1

    return (x, y)


def save_checkpoint(model, optimizer, iteration, out):
    obj = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(obj, out)
    print(f"Wrote {out}")


def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
