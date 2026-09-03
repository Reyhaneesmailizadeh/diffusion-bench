"""
Generate samples and create result figures.
Per dataset, one figure per loss_type: 3 pred_types (rows) x 6 cols (GT + 5 dims).
Output: 6 figures (2 datasets x 3 loss types).
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
DIMS = [2, 8, 32]
PRED_TYPES = ["x", "v"]
LOSS_TYPES = ["x", "v"]

N_SAMPLES = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ground_truth_2d(dataset):
    ds = ToyDiffusionDataset(dataset, dim=2)
    return ds.data.numpy()


def generate_samples(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except Exception:
        return None

    config = ckpt["config"]
    model = MLPDenoiser(
        data_dim=config["data_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    flow = FlowMatching(sample_steps=config["sample_steps"]).to(DEVICE)
    with torch.no_grad():
        samples = flow.sample(model, shape=(N_SAMPLES, config["data_dim"]),
                              pred_type=config["pred_type"], device=DEVICE)
    return samples.cpu().numpy()


def samples_to_2d(samples, dataset, dim):
    ds = ToyDiffusionDataset(dataset, dim=dim)
    return ds.to_2d(samples)


def make_figure(dataset, loss_type, checkpoint_dir, assets_dir):
    gt_2d = load_ground_truth_2d(dataset)

    n_cols = 1 + len(DIMS)
    n_rows = len(PRED_TYPES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    title = dataset.replace("_", " ").title()
    fig.suptitle(f"{title} — {loss_type}-loss", fontsize=16, fontweight="bold", y=0.995)

    pad = 0.15
    xlim = (gt_2d[:, 0].min() - pad, gt_2d[:, 0].max() + pad)
    ylim = (gt_2d[:, 1].min() - pad, gt_2d[:, 1].max() + pad)

    for row, pred_type in enumerate(PRED_TYPES):
        ax = axes[row, 0]
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=2, alpha=0.4, c="black", rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_ylabel(f"{pred_type}-pred", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title("Ground Truth", fontsize=11, fontweight="bold")

        for col, dim in enumerate(DIMS):
            ax = axes[row, 1 + col]
            ckpt = checkpoint_dir / dataset / f"dim{dim}" / f"{pred_type}_pred_{loss_type}_loss" / "checkpoint.pt"

            if ckpt.exists():
                samples = generate_samples(ckpt)
                if samples is not None:
                    s2d = samples_to_2d(samples, dataset, dim)
                    valid = np.isfinite(s2d).all(axis=1)
                    if valid.sum() >= 10:
                        ax.scatter(s2d[valid][:, 0], s2d[valid][:, 1], s=2, alpha=0.4,
                                   c="#2563eb", rasterized=True)
                    else:
                        ax.text(0.5, 0.5, "DIVERGED", ha="center", va="center",
                                transform=ax.transAxes, color="red", fontweight="bold")
                        ax.set_facecolor("#fff5f5")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"D={dim}", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = assets_dir / f"part2_{dataset}_{loss_type}_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--assets-dir", type=str, default="assets/experiments")
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()

    global N_SAMPLES
    N_SAMPLES = args.n_samples

    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}, Samples: {N_SAMPLES}\n")

    for dataset in DATASETS:
        for loss_type in LOSS_TYPES:
            print(f"=== {dataset} / {loss_type}-loss ===")
            make_figure(dataset, loss_type, Path(args.checkpoint_dir), assets_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
