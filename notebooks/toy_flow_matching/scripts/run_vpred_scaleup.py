"""
Part 3: Focus on v-pred at D=32. Can we make it work by scaling up model capacity?
Sweep hidden_dim at fixed 25K training steps. x-pred baseline for reference.
"""

import argparse
import time
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataloader import get_dataloader
from src.diffusion import FlowMatching
from src.model import MLPDenoiser

DIM = 32
LOSS_TYPE = "v"
DATASETS = ["swiss_roll"]
BATCH_SIZE = 1024
LR = 1e-3
N_LAYERS = 5
TRAIN_STEPS = 25000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIMS = [64, 128, 256, 512, 1024]


def train_one(dataset, pred_type, hidden_dim):
    dl = get_dataloader(name=dataset, dim=DIM, batch_size=BATCH_SIZE)
    flow = FlowMatching(sample_steps=50).to(DEVICE)
    model = MLPDenoiser(DIM, hidden_dim=hidden_dim, n_layers=N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    step = 0
    while step < TRAIN_STEPS:
        for batch in dl:
            if step >= TRAIN_STEPS:
                break
            x1 = batch.to(DEVICE)
            loss = flow.training_losses(model, x1, pred_type, LOSS_TYPE)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
    return model, losses[-1], losses, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    args = parser.parse_args()

    save_dir = Path("checkpoints_scaleup")
    save_dir.mkdir(exist_ok=True)
    datasets = [args.dataset] if args.dataset else DATASETS

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"{dataset} / D={DIM} / v-loss / {TRAIN_STEPS} steps")
        print(f"{'='*60}")

        for hidden_dim in HIDDEN_DIMS:
            for pred_type in ["x", "v"]:
                tag = f"{dataset}_h{hidden_dim}_{pred_type}"
                ckpt = save_dir / f"{tag}.pt"
                if ckpt.exists():
                    print(f"  SKIP {tag}")
                    continue

                t0 = time.time()
                model, final_loss, all_losses, n_params = train_one(
                    dataset, pred_type, hidden_dim)
                elapsed = time.time() - t0
                print(f"  {tag}: loss={final_loss:.6f} params={n_params:,} ({elapsed:.0f}s)")

                torch.save({
                    "dataset": dataset, "pred_type": pred_type,
                    "hidden_dim": hidden_dim, "train_steps": TRAIN_STEPS,
                    "final_loss": final_loss, "losses": all_losses,
                    "n_params": n_params,
                    "model_state_dict": model.state_dict(),
                    "config": {"data_dim": DIM, "hidden_dim": hidden_dim,
                               "n_layers": N_LAYERS, "sample_steps": 50,
                               "pred_type": pred_type, "loss_type": LOSS_TYPE,
                               "dataset": dataset},
                }, ckpt)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: D={DIM}, v-loss, {TRAIN_STEPS} steps")
    print(f"{'='*70}")
    print(f"{'hidden':>8s} | {'params':>10s} | {'x-pred':>10s} | {'v-pred':>10s} | {'ratio':>8s}")
    print("-" * 55)
    for h in HIDDEN_DIMS:
        x_losses, v_losses = [], []
        n_p = 0
        for ds in datasets:
            xp = save_dir / f"{ds}_h{h}_x.pt"
            vp = save_dir / f"{ds}_h{h}_v.pt"
            if xp.exists():
                d = torch.load(xp, map_location="cpu", weights_only=False)
                x_losses.append(d["final_loss"]); n_p = d["n_params"]
            if vp.exists():
                d = torch.load(vp, map_location="cpu", weights_only=False)
                v_losses.append(d["final_loss"])
        if x_losses and v_losses:
            xm, vm = sum(x_losses)/len(x_losses), sum(v_losses)/len(v_losses)
            print(f"{h:>8d} | {n_p:>10,} | {xm:>10.4f} | {vm:>10.4f} | {vm/xm:>8.1f}x")


if __name__ == "__main__":
    main()
