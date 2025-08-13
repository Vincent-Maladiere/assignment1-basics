import json
import typer
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    cross_entropy,
    AdamW,
    save_checkpoint,
    get_batch,
    learning_rate_schedule,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def _prepare_dirs(path_dir_output):
    str_date = f"{datetime.now()}".replace(" ", "_")
    path_dir_output = Path(path_dir_output) / str_date

    path_dir_checkpoints = path_dir_output / "checkpoints"
    path_dir_checkpoints.mkdir(parents=True, exist_ok=True)

    path_dir_stats = path_dir_output
    path_dir_stats.mkdir(parents=True, exist_ok=True)

    return {
        "checkpoints": path_dir_checkpoints,
        "stats": path_dir_stats,
    }


def _load_dataset_mmap(path_dir, index):
    if not (path_dir := Path(path_dir)).exists():
        raise OSError(f"{path_dir} doesn't exist")

    paths = list(path_dir.glob("*.npy"))
    path = paths[index % len(paths)]

    return np.load(path, mmap_mode="r")


def _apply_lr_schedule(optimizer, t, n_iter, lr_min, lr_max):
    lr = learning_rate_schedule(
        t,
        lr_max=lr_max,
        lr_min=lr_min,
        T_w=int(n_iter * 0.10),
        T_c=int(n_iter * 0.75),
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _to_tensor(dataset, indices, device):
    indices_np = indices.cpu().numpy()
    # mps doesn't support uint16 tensors, and the Embedding module
    # can't deal with int16.
    x = dataset[indices_np].astype("int32")
    return torch.as_tensor(x, device=device)


def _save_run_logs(run_logs, path):
    path.write_text(json.dumps(run_logs))
    print(f"Wrote {path}")


def main(
    path_dir_dataset: str = "data/tiny_stories/ids",
    path_dir_output: str = "data/tiny_stories/out",
    n_iter: int = 5000,
    batch_size: int = 32,
    context_length: int = 256,
    vocab_size: int = 2**16,
    d_model: int = 512,
    d_ff: int = 1344,
    n_layers: int = 4,
    n_heads: int = 16,
    rope_theta: int = 10000,
    lr: float = 1e-3,
    lr_min: float = 1e-4,
    lr_max: float = 1e-1,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    max_l2_norm: float | None = None,
    device: str = "cpu",
):
    run_logs = {"params": locals()}
    ### DEBUG
    import random

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ###

    paths = _prepare_dirs(path_dir_output)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=n_layers,
        num_heads=n_heads,
        rope_theta=rope_theta,
        device=device,
    ).train()
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=1e-8,
        # max_l2_norm=max_l2_norm,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=1e-5)
    step_results = []
    for i in range(1, n_iter + 1):

        dataset = _load_dataset_mmap(path_dir_dataset, i)
        # _apply_lr_schedule(optimizer, i, n_iter, lr_min, lr_max)
        optimizer.zero_grad()
        x_indices, y_indices = get_batch(
            dataset,
            batch_size,
            context_length,
            device,
        )
        x = _to_tensor(dataset, x_indices, device)
        y = _to_tensor(dataset, y_indices, device)

        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(i, loss)
        step_results.append({"loss": float(loss.item())})

        if i > 0 and i % 5 == 0:
            save_checkpoint(model, optimizer, i, paths["checkpoints"] / f"{i}.pt")

    run_logs["step_results"] = step_results
    _save_run_logs(run_logs, paths["stats"] / "stats.json")
    save_checkpoint(model, optimizer, i, paths["checkpoints"] / "final.pt")


if __name__ == "__main__":
    typer.run(main)
