"""
Train flow matching models.
Adam, no grad clip, 25000 training steps, batch 1024.
"""

from pathlib import Path

import torch
torch.set_float32_matmul_precision('high')

from src.dataloader import get_dataloader
from src.diffusion import FlowMatching
from src.model import MLPDenoiser

PRED_TYPES = ["x", "v"]
LOSS_TYPES = ["x", "v"]


def train_one(
    pred_type: str,
    loss_type: str,
    dataset: str = "swiss_roll",
    dim: int = 2,
    train_steps: int = 25000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    n_layers: int = 5,
    sample_steps: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str | Path | None = None,
    milestone_steps: list[int] | None = None,
) -> tuple[object, list[float]]:
    tag = f"{pred_type}_pred_{loss_type}_loss"
    print(f"\n  Training: {tag}  (dim={dim})")

    dl = get_dataloader(name=dataset, dim=dim, batch_size=batch_size)
    flow = FlowMatching(sample_steps=sample_steps).to(device)
    flow.training_losses = torch.compile(flow.training_losses)
    model = MLPDenoiser(dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    milestones = set(milestone_steps or [])
    config = {
        "dataset": dataset,
        "pred_type": pred_type,
        "loss_type": loss_type,
        "data_dim": dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "sample_steps": sample_steps,
    }

    losses = []
    step = 0
    while step < train_steps:
        for batch in dl:
            if step >= train_steps:
                break
            x1 = batch.to(device)
            loss = flow.training_losses(model, x1, pred_type, loss_type)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 500 == 0:
                print(f"    step {step}/{train_steps}  loss={loss.item():.6f}")

            if save_dir is not None and step in milestones:
                out = Path(save_dir) / tag
                out.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": config,
                     "step": step, "loss": loss.item()},
                    out / f"checkpoint_step{step}.pt",
                )

    if save_dir is not None:
        out = Path(save_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config,
             "final_loss": losses[-1], "losses": losses},
            out / "checkpoint.pt",
        )

    return model, losses
