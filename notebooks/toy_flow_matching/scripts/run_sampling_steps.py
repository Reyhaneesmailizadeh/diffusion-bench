"""
Part 3a: How does x-pred (v-loss, D=32) sample quality degrade with fewer Euler steps?
Uses trained checkpoints from Part 1.
One figure: 3 dataset rows × (GT + step counts) columns.
Highlights the "good" (>=20 steps) vs "degraded" (<10 steps) regime.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataloader import ToyDiffusionDataset
from src.diffusion import FlowMatching
from src.model import MLPDenoiser

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIM = 32
STEP_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200]
PRED_TYPE = "x"
LOSS_TYPE = "v"
N_SAMPLES = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("checkpoints")
ASSETS_DIR = Path("assets/experiments")

GOOD_COLOR = "#2563eb"
BAD_COLOR = "#94a3b8"  # muted for degraded samples


def load_model(dataset):
    ckpt_path = CKPT_DIR / dataset / f"dim{DIM}" / f"{PRED_TYPE}_pred_{LOSS_TYPE}_loss" / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt["config"]
    model = MLPDenoiser(
        data_dim=config["data_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(DATASETS)
    n_cols = 1 + len(STEP_COUNTS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows))
    fig.suptitle(f"Part 3: x-pred Euler Sampling at D={DIM} — How Many Steps Do We Need?",
                 fontsize=14, fontweight="bold", y=1.01)

    for row, dataset in enumerate(DATASETS):
        ds_2d = ToyDiffusionDataset(dataset, dim=2)
        ds_dim = ToyDiffusionDataset(dataset, dim=DIM)
        gt_2d = ds_2d.data.numpy()[:N_SAMPLES]
        pad = 0.15
        xlim = (gt_2d[:, 0].min() - pad, gt_2d[:, 0].max() + pad)
        ylim = (gt_2d[:, 1].min() - pad, gt_2d[:, 1].max() + pad)

        # GT column
        ax = axes[row, 0]
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=1.5, alpha=0.35, c="black", rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(dataset.replace("_", " ").title(), fontsize=11, fontweight="bold")
        if row == 0:
            ax.set_title("GT", fontsize=10, fontweight="bold")

        # Sample columns
        model = load_model(dataset)
        for col, steps in enumerate(STEP_COUNTS):
            ax = axes[row, 1 + col]

            flow = FlowMatching(sample_steps=steps)
            with torch.no_grad():
                samples = flow.sample(model, (N_SAMPLES, DIM), PRED_TYPE, DEVICE).cpu().numpy()
            s2d = ds_dim.to_2d(samples)
            valid = np.isfinite(s2d).all(axis=1)

            color = GOOD_COLOR if steps >= 20 else BAD_COLOR
            if valid.sum() >= 10:
                ax.scatter(s2d[valid, 0], s2d[valid, 1], s=1.5, alpha=0.35,
                           c=color, rasterized=True)
            else:
                ax.text(0.5, 0.5, "DIVERGED", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontweight="bold")
                ax.set_facecolor("#fff5f5")

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{steps} steps", fontsize=10, fontweight="bold",
                             color=color)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / "part4_sampling_steps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
