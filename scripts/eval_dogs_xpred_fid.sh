#!/bin/bash
# Compute FID for dogs-lightningdit-xpred at steps 17k, 34k, 50k.
# Each step maps to an epoch (100 steps/epoch): 17k=ep170, 34k=ep340, 50k=ep500.
# Results are appended to results/evals/stage2/training/dogs/dogs-lightningdit-xpred_ema.csv

set -e

EXPERIMENT_NAME="dogs-lightningdit-xpred"
CONFIG="configs/stage2/eval/dogs-lightningdit-xpred-fid.yaml"
CKPT_DIR="ckpts/${EXPERIMENT_NAME}/checkpoints"
GPUS=${GPUS:-1}

for epoch in 170 340 500; do
    ckpt="${CKPT_DIR}/ep-$(printf '%07d' ${epoch}).pt"
    if [ ! -f "${ckpt}" ]; then
        echo "Checkpoint not found, skipping: ${ckpt}"
        continue
    fi
    echo "Evaluating epoch ${epoch} ($(( epoch * 100 / 1000 ))k steps): ${ckpt}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" torchrun --nproc_per_node=${GPUS} \
        src/offline_eval.py \
        --config "${CONFIG}" \
        --checkpoint "${ckpt}"
done

echo "Done. Results written to results/evals/stage2/training/dogs/${EXPERIMENT_NAME}_ema.csv"
