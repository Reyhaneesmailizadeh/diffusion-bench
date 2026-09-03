"""
Part 0 figures:
1. Data visualization: 3 datasets at D=2, 32 projected back to 2D
2. V-pred warm-up: train v-pred/v-loss on D=2 for all 3 datasets
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
from src.train import train_one

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASSETS_DIR = Path("assets/experiments")
DATASETS = ["swiss_roll", "gaussians", "circles"]
N_SAMPLES = 2000


def plot_data_check():
    """Data at D=2, 32 projected back to 2D."""
    dims = [2, 32]
    fig, axes = plt.subplots(len(DATASETS), len(dims),
                              figsize=(3.5 * len(dims), 3.5 * len(DATASETS)))
    fig.suptitle("Part 0: Data Visualization (projected to 2D)",
                 fontsize=14, fontweight="bold", y=1.0)

    for row, ds_name in enumerate(DATASETS):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            ds = ToyDiffusionDataset(ds_name, dim=dim)
            data_2d = ds.to_2d(ds.data.numpy()[:N_SAMPLES])
            ax.scatter(data_2d[:, 0], data_2d[:, 1], s=2, alpha=0.4,
                       c="black", rasterized=True)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"D = {dim}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ds_name.replace("_", " ").title(),
                              fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / "part1_data.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_vpred_warmup():
    """V-pred on D=2: GT vs generated samples for all 3 datasets."""
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(7, 3.5 * len(DATASETS)))
    fig.suptitle("Part 1: v-pred Warm-up (D=2, 25K steps)",
                 fontsize=14, fontweight="bold", y=1.0)

    for row, ds_name in enumerate(DATASETS):
        ds = ToyDiffusionDataset(ds_name, dim=2)
        gt = ds.data.numpy()[:N_SAMPLES]
        pad = 0.15
        xlim = (gt[:, 0].min() - pad, gt[:, 0].max() + pad)
        ylim = (gt[:, 1].min() - pad, gt[:, 1].max() + pad)

        model, losses = train_one(
            pred_type="v", loss_type="v", dataset=ds_name, dim=2,
            train_steps=25000, device=DEVICE,
            save_dir=str(Path("checkpoints_part0") / ds_name / "dim2"),
        )

        flow = FlowMatching(sample_steps=50)
        with torch.no_grad():
            samples = flow.sample(model, (N_SAMPLES, 2), "v", DEVICE).cpu().numpy()

        # GT
        ax = axes[row, 0]
        ax.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.4, c="black", rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title("Ground Truth", fontsize=12, fontweight="bold")
        ax.set_ylabel(ds_name.replace("_", " ").title(),
                      fontsize=11, fontweight="bold")

        # Samples
        ax = axes[row, 1]
        ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.4,
                   c="#2563eb", rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title("v-pred Samples", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / "part1_vpred.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plot_data_check()
    plot_vpred_warmup()
