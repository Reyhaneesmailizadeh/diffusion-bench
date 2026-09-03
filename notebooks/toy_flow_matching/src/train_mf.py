"""
Train MeanFlow on toy datasets at D=32.
Uses torch.compile for faster JVP training.
"""

import argparse
import time
from pathlib import Path

import torch
torch.set_float32_matmul_precision('high')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataloader import get_dataloader
from src.diffusion_mf import MeanFlow
from src.model_mf import MLPDenoiserMF

DATASETS = ["swiss_roll", "gaussians", "circles"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one(dataset, dim, train_steps=25000, hidden_dim=256, n_layers=5,
              lr=1e-3, batch_size=1024):
    dl = get_dataloader(name=dataset, dim=dim, batch_size=batch_size)
    mf = MeanFlow(fm_ratio=0.5).to(DEVICE)
    mf.training_losses = torch.compile(mf.training_losses)
    model = MLPDenoiserMF(dim, hidden_dim=hidden_dim, n_layers=n_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    metrics_log = {"mse_fm": [], "mse_mf": [], "dudt_norm": []}
    step = 0
    while step < train_steps:
        for batch in dl:
            if step >= train_steps:
                break
            x1 = batch.to(DEVICE)
            loss, mse_fm, mse_mf, dudt_n = mf.training_losses(model, x1)
            losses.append(loss.item())
            metrics_log["mse_fm"].append(mse_fm.item())
            metrics_log["mse_mf"].append(mse_mf.item())
            metrics_log["dudt_norm"].append(dudt_n.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % 10000 == 0:
                n = min(1000, len(losses))
                avg = {k: sum(v[-n:]) / n for k, v in metrics_log.items()}
                print(f"    step {step}/{train_steps}  "
                      f"mse_fm={avg['mse_fm']:.6f}  "
                      f"mse_mf={avg['mse_mf']:.6f}  "
                      f"dudt={avg['dudt_norm']:.4f}")

    return model, losses, metrics_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    args = parser.parse_args()

    save_dir = Path("checkpoints_meanflow")
    save_dir.mkdir(exist_ok=True)
    datasets = [args.dataset] if args.dataset else DATASETS

    for dataset in datasets:
        tag = f"{dataset}_dim{args.dim}"
        ckpt_path = save_dir / f"{tag}.pt"
        if ckpt_path.exists():
            print(f"SKIP {tag}")
            continue

        print(f"\n{'='*50}")
        print(f"MeanFlow: {dataset}, D={args.dim}")
        print(f"{'='*50}")

        t0 = time.time()
        model, losses, metrics_log = train_one(
            dataset, args.dim,
            train_steps=args.train_steps,
            hidden_dim=args.hidden_dim,
        )
        elapsed = time.time() - t0
        print(f"  Done: mse_mf={metrics_log['mse_mf'][-1]:.6f} ({elapsed:.0f}s)")

        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "data_dim": args.dim, "hidden_dim": args.hidden_dim,
                "n_layers": 5, "dataset": dataset,
            },
            "final_loss": losses[-1],
            "metrics_log": metrics_log,
            "losses": losses,
        }, ckpt_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()
