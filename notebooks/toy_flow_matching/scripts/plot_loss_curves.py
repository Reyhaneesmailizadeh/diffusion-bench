"""
Plot training loss curves for all pred/loss combinations.
One figure per dataset: 2x2 grid (x/v pred x x/v loss), all dims overlaid.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIMS = [2, 8, 32]
PRED_TYPES = ["x", "v"]
LOSS_TYPES = ["x", "v"]
CKPT_DIR = Path("checkpoints")
ASSETS_DIR = Path("assets/experiments")

DIM_COLORS = {2: "#2563eb", 8: "#16a34a", 32: "#dc2626"}


def smooth(losses, window=50):
    """Simple moving average."""
    if len(losses) < window:
        return losses
    kernel = np.ones(window) / window
    return np.convolve(losses, kernel, mode="valid")


def make_figure(dataset):
    fig, axes = plt.subplots(len(PRED_TYPES), len(LOSS_TYPES),
                              figsize=(5 * len(LOSS_TYPES), 4 * len(PRED_TYPES)),
                              sharex=True)
    title = dataset.replace("_", " ").title()
    fig.suptitle(f"{title} - Training Loss Curves", fontsize=14, fontweight="bold")

    for row, pred_type in enumerate(PRED_TYPES):
        for col, loss_type in enumerate(LOSS_TYPES):
            ax = axes[row, col]
            ax.set_title(f"{pred_type}-pred / {loss_type}-loss", fontsize=11)

            for dim in DIMS:
                ckpt_path = (CKPT_DIR / dataset / f"dim{dim}" /
                             f"{pred_type}_pred_{loss_type}_loss" / "checkpoint.pt")
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                losses = ckpt.get("losses", [])
                if not losses:
                    continue
                smoothed = smooth(losses)
                ax.plot(smoothed, label=f"D={dim}", color=DIM_COLORS[dim],
                        alpha=0.8, linewidth=1.2)

            ax.set_yscale("log")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = ASSETS_DIR / f"part2_loss_curves_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        make_figure(dataset)


if __name__ == "__main__":
    main()
