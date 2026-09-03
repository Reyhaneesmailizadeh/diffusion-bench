"""
Run all experiments: 3 datasets x 3 dims x 2 pred_types x 2 loss_types = 36.
Supports --dataset and --loss-type flags for parallel jobs.
"""

import argparse
import time
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train_one, PRED_TYPES, LOSS_TYPES

DATASETS = ["swiss_roll", "gaussians", "circles"]
DIMS = [2, 8, 32]
MILESTONES = [500, 1000, 5000, 10000, 25000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=25000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS)
    parser.add_argument("--loss-type", type=str, default=None, choices=LOSS_TYPES)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    datasets = [args.dataset] if args.dataset else DATASETS
    loss_types = [args.loss_type] if args.loss_type else LOSS_TYPES

    for dataset in datasets:
        for loss_type in loss_types:
            print(f"\n{'='*60}")
            print(f"{dataset} / {loss_type}-loss")
            print(f"{'='*60}")

            for dim in DIMS:
                print(f"\n--- dim={dim} ---")
                save_sub = checkpoint_dir / dataset / f"dim{dim}"

                for pred_type in PRED_TYPES:
                    tag = f"{pred_type}_pred_{loss_type}_loss"
                    final_ckpt = save_sub / tag / "checkpoint.pt"

                    if final_ckpt.exists():
                        print(f"  SKIP {tag}")
                        continue

                    t0 = time.time()
                    model, losses = train_one(
                        pred_type=pred_type, loss_type=loss_type,
                        dataset=dataset, dim=dim, train_steps=args.train_steps,
                        save_dir=str(save_sub),
                        milestone_steps=[m for m in MILESTONES if m <= args.train_steps],
                    )
                    print(f"    -> {losses[-1]:.6f} ({time.time()-t0:.0f}s)")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print("\nDONE.")


if __name__ == "__main__":
    main()
