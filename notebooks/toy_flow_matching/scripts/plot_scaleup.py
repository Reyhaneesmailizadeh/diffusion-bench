"""
Visualize Part 3: v-pred scaleup at D=32.
Single column: rows = [GT, x-pred(256), v-pred(64..1024)].
All at 25K training steps.
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

DIM = 32
HIDDEN_DIMS = [64, 128, 256, 512, 1024]
DATASETS = ["swiss_roll"]
N_SAMPLES = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("checkpoints_scaleup")
ASSETS_DIR = Path("assets/experiments")


def generate_samples(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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
        samples = flow.sample(model, (N_SAMPLES, config["data_dim"]),
                              config["pred_type"], DEVICE)
    return samples.cpu().numpy(), ckpt["final_loss"]


def make_figure(dataset):
    ds = ToyDiffusionDataset(dataset, dim=DIM)
    gt_2d = ds.to_2d(ds.data.numpy()[:N_SAMPLES])

    pad = 0.15
    xlim = (gt_2d[:, 0].min() - pad, gt_2d[:, 0].max() + pad)
    ylim = (gt_2d[:, 1].min() - pad, gt_2d[:, 1].max() + pad)

    param_counts = {64: "29K", 128: "91K", 256: "313K", 512: "1.1M", 1024: "4.4M"}
    row_specs = [
        ("Ground Truth", None, None, "black"),
        ("x-pred (h=256, 313K)", "x", 256, "#2563eb"),
    ] + [(f"v-pred (h={h}, {param_counts.get(h, '?')})", "v", h, "#dc2626") for h in HIDDEN_DIMS]

    n_cols = len(row_specs)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5))
    title = dataset.replace("_", " ").title()
    fig.suptitle(f"{title} - v-pred capacity scaling at D={DIM} (25K steps)",
                 fontsize=13, fontweight="bold", y=1.02)

    for col, (label, pred_type, hidden, color) in enumerate(row_specs):
        ax = axes[col]

        if col == 0:
            ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=2, alpha=0.4,
                       c=color, rasterized=True)
        else:
            ckpt = CKPT_DIR / f"{dataset}_h{hidden}_{pred_type}.pt"
            if ckpt.exists():
                samples, loss = generate_samples(ckpt)
                s2d = ds.to_2d(samples)
                valid = np.isfinite(s2d).all(axis=1)
                if valid.sum() >= 10:
                    ax.scatter(s2d[valid, 0], s2d[valid, 1], s=2, alpha=0.4,
                               c=color, rasterized=True)
                    ax.text(0.02, 0.98, f"L={loss:.4f}", transform=ax.transAxes,
                            fontsize=7, va="top", ha="left", alpha=0.8)
                else:
                    ax.text(0.5, 0.5, "DIVERGED", ha="center", va="center",
                            transform=ax.transAxes, color="red", fontweight="bold")

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / f"part3_scaleup_{dataset}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        make_figure(dataset)


if __name__ == "__main__":
    main()
