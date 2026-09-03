"""
Generate swiss_roll, gaussians, and circles 2D data, project to dims 2, 8, 32.
No normalization — raw projected data.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import make_swiss_roll, make_circles

DIMS = [8, 32]
DATASETS = ["swiss_roll", "gaussians", "circles"]


def random_orthogonal_projection(d_low, d_high, seed):
    rng = np.random.RandomState(seed)
    A = rng.randn(d_high, d_low)
    Q, _ = np.linalg.qr(A)
    return Q[:, :d_low].T  # (d_low, d_high)


def make_ring_gaussians(n_samples, n_modes=8, std=0.05, seed=42):
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    labels = rng.randint(0, n_modes, size=n_samples)
    points = centers[labels] + rng.randn(n_samples, 2) * std
    return points, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "data"
    out.mkdir(parents=True, exist_ok=True)

    # Shared projection matrices per dim
    projections = {d: random_orthogonal_projection(2, d, seed=args.seed + d) for d in DIMS}

    # Swiss roll: [:, [0,2]] / 10.0 (matches notebook)
    X, _ = make_swiss_roll(args.n_samples, random_state=args.seed)
    swiss_2d = X[:, [0, 2]] / 10.0

    save_dict = {"2d": swiss_2d.astype(np.float32)}
    for d, P in projections.items():
        save_dict[f"{d}d"] = (swiss_2d @ P).astype(np.float32)
        save_dict[f"P_{d}"] = P.astype(np.float32)
    np.savez(out / "swiss_roll.npz", **save_dict)
    print(f"swiss_roll: 2D {swiss_2d.shape} -> {', '.join(f'{d}D' for d in DIMS)}")

    # 8-mode ring of Gaussians
    gauss_2d, labels = make_ring_gaussians(args.n_samples, seed=args.seed)

    save_dict = {"2d": gauss_2d.astype(np.float32), "labels": labels.astype(np.int64)}
    for d, P in projections.items():
        save_dict[f"{d}d"] = (gauss_2d @ P).astype(np.float32)
        save_dict[f"P_{d}"] = P.astype(np.float32)
    np.savez(out / "gaussians.npz", **save_dict)
    print(f"gaussians:  2D {gauss_2d.shape} -> {', '.join(f'{d}D' for d in DIMS)}")

    # Two concentric circles
    circles_2d, labels = make_circles(args.n_samples, noise=0.05, factor=0.5,
                                       random_state=args.seed)

    save_dict = {"2d": circles_2d.astype(np.float32), "labels": labels.astype(np.int64)}
    for d, P in projections.items():
        save_dict[f"{d}d"] = (circles_2d @ P).astype(np.float32)
        save_dict[f"P_{d}"] = P.astype(np.float32)
    np.savez(out / "circles.npz", **save_dict)
    print(f"circles:    2D {circles_2d.shape} -> {', '.join(f'{d}D' for d in DIMS)}")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
