#!/bin/bash
# =============================================================================
# Sinkhorn+Drifting aligned training on ImageNet-256
#
# Architecture: LightningDiT (drifting-aligned)
#   - patch_size=2, 256 tokens, 134M params
#   - QK-Norm + RoPE + SwiGLU + RMSNorm + 16 cls tokens
#   - noise_classes=64, noise_coords=32
#   - cfg_embedder (uniform CFG [1.0, 4.0], 50% dropout)
#
# Loss:  Sinkhorn coupling + split drift-form (sinkhorn improvement)
# Steps: ~30000 (96 epochs × ~313 iters/epoch)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_NAME="dit_b2_sinkhorn_drift_aligned_${TIMESTAMP}"

# ── Hyperparams ──────────────────────────────────────────────────────────────
COUPLING="sinkhorn"
DRIFT_FORM="split"
SINKHORN_ITERS=20
SINKHORN_MARGINAL="weighted_cols"
BACKBONE="dit_b_2"       # patch_size=2, 256 tokens
TEMPS="0.02 0.05 0.2"
EPOCHS=96                # ~30000 steps with default iters_per_epoch
LR=2e-4
WEIGHT_DECAY=0.01
WARMUP=5000
EMA_DECAY=0.999
NNEG=64                  # gen_per_label=64 (matches drifting)
NPOS=64                  # pos_per_sample=64 (matches drifting)
NUNCOND=16               # neg_per_sample=16 (matches drifting)
CLASSES_PER_STEP=64      # train_batch_size=64 (matches drifting)
NOISE_CLASSES=64         # drifting-aligned
NOISE_COORDS=32          # drifting-aligned

# ── Logging ──────────────────────────────────────────────────────────────────
TB_PORT=6006
RUN_DIR="${PROJECT_DIR}/runs/imagenet256_drift/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

# ── Launch TensorBoard ──────────────────────────────────────────────────────
echo "Starting TensorBoard on port ${TB_PORT}..."
tensorboard --logdir "${PROJECT_DIR}/runs/imagenet256_drift" \
    --port "${TB_PORT}" --bind_all --reload_interval 30 &
TB_PID=$!
echo "TensorBoard → http://$(hostname):${TB_PORT} (PID ${TB_PID})"
cleanup() { kill ${TB_PID} 2>/dev/null || true; }
trap cleanup EXIT INT TERM
sleep 2

# ── Launch Training ─────────────────────────────────────────────────────────
echo "============================================"
echo "Sinkhorn+Drifting aligned training"
echo "  Backbone:     ${BACKBONE} (133M params, 256 tokens)"
echo "  Coupling:     ${COUPLING} + ${DRIFT_FORM}"
echo "  Steps:        ${EPOCHS} epochs ≈ 30000 step"
echo "  LR/WD/EMA:    ${LR} / ${WEIGHT_DECAY} / ${EMA_DECAY}"
echo "  Neg/Pos/Unc:  ${NNEG} / ${NPOS} / ${NUNCOND}"
echo "  Classes/step: ${CLASSES_PER_STEP}"
echo "  Noise:        ${NOISE_CLASSES} × ${NOISE_COORDS}"
echo "  Run dir:      ${RUN_DIR}"
echo "  TensorBoard:  http://$(hostname):${TB_PORT}"
echo "============================================"

cd "${PROJECT_DIR}"

torchrun --standalone --nproc_per_node=4 \
    -m imagenet.train_drifting \
    --cache-dir "/home/liushuai/data1/liujiawei/imagenet256/cache_full" \
    --backbone "${BACKBONE}" \
    --coupling "${COUPLING}" \
    --drift-form "${DRIFT_FORM}" \
    --sinkhorn-iters "${SINKHORN_ITERS}" \
    --sinkhorn-marginal "${SINKHORN_MARGINAL}" \
    --temps ${TEMPS} \
    --nneg "${NNEG}" \
    --npos "${NPOS}" \
    --nuncond "${NUNCOND}" \
    --classes-per-step "${CLASSES_PER_STEP}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-steps "${WARMUP}" \
    --ema-decay "${EMA_DECAY}" \
    --noise-classes "${NOISE_CLASSES}" \
    --noise-coords "${NOISE_COORDS}" \
    --feature-mode encoder \
    --encoder-every-n-blocks 2 \
    --gen-mode direct \
    --run-name "${RUN_NAME}" \
    --sample-every-epochs 0.8 \
    --decode-rgb-every-epochs 0.8 \
    --fid-every-epochs 5 \
    --fid-num-samples 10000 \
    --fid-ref-stats "/home/liushuai/data1/liujiawei/imagenet256/fid_ref/imagenet_256_fid_stats.npz" \
    --tensorboard \
    2>&1 | tee "${RUN_DIR}/train.log"

echo "Done! Checkpoints in ${RUN_DIR}/"
