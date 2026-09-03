"""
Part 3 MeanFlow figure: Compare flow matching (x-pred, multi-step) vs MeanFlow.
All at D=32. One figure: 3 dataset rows × (GT + RF steps + MF steps) columns.
Uniform axis limits per row, aligned grids.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataloader import ToyDiffusionDataset
from src.diffusion import FlowMatching
from src.diffusion_mf import MeanFlow
from src.model import MLPDenoiser
from src.model_mf import MLPDenoiserMF

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIM = 32
N_SAMPLES = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RF_CKPT_DIR = Path("checkpoints")
MF_CKPT_DIR = Path("checkpoints_meanflow")
ASSETS_DIR = Path("assets/experiments")

RF_COLOR = "#2563eb"
MF_COLOR = "#dc2626"

COLS = [
    ("GT",       None, None, "black"),
    ("50 steps", "rf", 50,   RF_COLOR),
    ("10 steps", "rf", 10,   RF_COLOR),
    ("5 steps",  "rf", 5,    RF_COLOR),
    ("1 step",   "rf", 1,    RF_COLOR),
    ("1 step",   "mf", 1,    MF_COLOR),
    ("2 steps",  "mf", 2,    MF_COLOR),
    ("5 steps",  "mf", 5,    MF_COLOR),
]


def load_rf_model(dataset):
    ckpt_path = RF_CKPT_DIR / dataset / f"dim{DIM}" / "x_pred_v_loss" / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = MLPDenoiser(cfg["data_dim"], cfg["hidden_dim"], cfg["n_layers"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_mf_model(dataset):
    ckpt_path = MF_CKPT_DIR / f"{dataset}_dim{DIM}.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = MLPDenoiserMF(cfg["data_dim"], cfg["hidden_dim"], cfg["n_layers"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def plot_panel(ax, pts, color, lim):
    if pts is not None:
        valid = np.isfinite(pts).all(axis=1)
        if valid.sum() >= 10:
            ax.scatter(pts[valid, 0], pts[valid, 1],
                       s=1.5, alpha=0.35, c=color, rasterized=True)
        else:
            ax.text(0.5, 0.5, "DIVERGED", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontweight="bold")
            ax.set_facecolor("#fff5f5")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(DATASETS)
    n_cols = len(COLS)
    cell_w, cell_h = 2.6, 2.6

    fig = plt.figure(figsize=(cell_w * n_cols + 0.8, cell_h * n_rows + 1.2))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.12,
                  left=0.06, right=0.99, top=0.88, bottom=0.02)

    # Group headers
    fig.text(0.065, 0.93, "GT", fontsize=10, fontweight="bold", ha="center")
    fig.text(0.37, 0.93, "Flow Matching (x-pred, v-loss)",
             fontsize=11, fontweight="bold", ha="center", color=RF_COLOR)
    fig.text(0.82, 0.93, "MeanFlow (x-pred)",
             fontsize=11, fontweight="bold", ha="center", color=MF_COLOR)
    fig.suptitle(f"Part 3: Flow Matching vs MeanFlow at D={DIM}",
                 fontsize=14, fontweight="bold", y=0.98)

    for row, dataset in enumerate(DATASETS):
        ds_2d = ToyDiffusionDataset(dataset, dim=2)
        ds_dim = ToyDiffusionDataset(dataset, dim=DIM)
        gt_2d = ds_2d.data.numpy()[:N_SAMPLES]

        # Uniform symmetric limit per row
        lim = float(np.abs(gt_2d).max()) + 0.15

        rf_model = load_rf_model(dataset)
        mf_model = load_mf_model(dataset)

        for col, (label, method, steps, color) in enumerate(COLS):
            ax = fig.add_subplot(gs[row, col])

            if method is None:
                plot_panel(ax, gt_2d, color, lim)
            elif method == "rf":
                flow = FlowMatching(sample_steps=steps)
                with torch.no_grad():
                    s = flow.sample(rf_model, (N_SAMPLES, DIM), "x", DEVICE).cpu().numpy()
                plot_panel(ax, ds_dim.to_2d(s), color, lim)
            elif method == "mf":
                mf = MeanFlow()
                with torch.no_grad():
                    s = mf.sample(mf_model, (N_SAMPLES, DIM), DEVICE, num_steps=steps).cpu().numpy()
                plot_panel(ax, ds_dim.to_2d(s), color, lim)

            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", color=color)
            if col == 0:
                ax.set_ylabel(dataset.replace("_", " ").title(),
                              fontsize=10, fontweight="bold")

            # Separator line between RF and MF
            if col == 4:
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_color("#aaa")
                ax.spines["right"].set_linewidth(1.5)

    out = ASSETS_DIR / "part4_meanflow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
