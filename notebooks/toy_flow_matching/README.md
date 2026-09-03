# Flow Matching Parameterization Study

Reproducing and extending the key finding from *"Back to Basics: Let Denoising Generative Models Denoise"* ([arxiv 2511.13720](https://arxiv.org/abs/2511.13720)).

See [`docs/assignment.md`](docs/assignment.md) for the full assignment.

## Quick Start

```bash
# install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# download data
uv run hf download xingjianleng/toy-data --local-dir data --repo-type dataset
```

## Running Experiments

### Part 0: Warm-up (data check + v-pred on D=2)

```bash
# trains v-pred/v-loss on D=2 for all 3 datasets, saves checkpoints + figures
uv run python scripts/plot_part0.py
```

### Part 1: Baseline (all parameterizations)

```bash
# train all 135 models (3 pred × 3 loss × 5 dims × 3 datasets)
uv run python scripts/run_all.py --dataset swiss_roll
uv run python scripts/run_all.py --dataset gaussians
uv run python scripts/run_all.py --dataset circles

# generate figures
uv run python scripts/sample_and_visualize.py
```

### Part 2: v-pred scaleup at D=32

```bash
# sweep hidden_dim × train_steps (90 models)
uv run python scripts/run_vpred_scaleup.py --dataset swiss_roll
uv run python scripts/run_vpred_scaleup.py --dataset gaussians
uv run python scripts/run_vpred_scaleup.py --dataset circles

# generate figures
uv run python scripts/plot_scaleup.py
```

### Part 3: MeanFlow one-step generation at D=128

```bash
# sampling steps comparison (uses Part 1 checkpoints)
uv run python scripts/run_sampling_steps.py

# train MeanFlow (25K steps, ~5 min/model)
uv run python src/train_mf.py

# generate comparison figure
uv run python scripts/plot_meanflow.py
```

## References

- [Back to Basics (JiT)](https://arxiv.org/abs/2511.13720): prediction parameterization in flow matching
- [RAE](https://arxiv.org/abs/2510.11690): parameterization and dimension
- [MeanFlow](https://arxiv.org/abs/2505.13447): one-step generation
- [DiT](https://github.com/facebookresearch/DiT): sinusoidal time embedding reference
- [SiT](https://github.com/willisma/SiT): Euler ODE sampling reference
