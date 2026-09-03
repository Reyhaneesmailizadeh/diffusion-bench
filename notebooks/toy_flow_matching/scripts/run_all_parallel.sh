#!/bin/bash
# Run all experiments in parallel, then generate figures from checkpoints.

set -e

echo "=== Stage 1: Training (parallel) ==="

uv run python scripts/plot_part0.py &
PID_P0=$!

uv run python scripts/run_all.py &
PID_P1=$!

uv run python scripts/run_vpred_scaleup.py &
PID_P2=$!

uv run python src/train_mf.py &
PID_MF=$!

echo "Waiting for all training jobs..."
echo "  Part 0  (PID $PID_P0)"
echo "  Part 1  (PID $PID_P1)"
echo "  Part 2  (PID $PID_P2)"
echo "  MeanFlow (PID $PID_MF)"

wait $PID_P0 && echo "Part 0 done." || echo "Part 0 FAILED."
wait $PID_P1 && echo "Part 1 done." || echo "Part 1 FAILED."
wait $PID_P2 && echo "Part 2 done." || echo "Part 2 FAILED."
wait $PID_MF && echo "MeanFlow done." || echo "MeanFlow FAILED."

echo ""
echo "=== Stage 2: Figures (parallel) ==="

uv run python scripts/sample_and_visualize.py &
uv run python scripts/plot_scaleup.py &
uv run python scripts/run_sampling_steps.py &
uv run python scripts/plot_meanflow.py &

wait
echo ""
echo "=== All done. ==="
