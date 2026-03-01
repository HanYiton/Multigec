#!/bin/bash
# End-to-end pipeline: Data Processing → SFT → GRPO → Evaluation
# Target hardware: 2x NVIDIA A30 (24GB each)
#
# Usage:
#   bash scripts/run_pipeline.sh
#
# Set these environment variables to control behavior:
#   MODEL_NAME     - Base model (default: Qwen/Qwen2.5-3B-Instruct)
#   SKIP_SFT       - Set to 1 to skip SFT stage
#   SKIP_GRPO      - Set to 1 to skip GRPO stage
#   WANDB_PROJECT  - W&B project name

set -e

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-3B-Instruct"}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"

echo "=============================================="
echo "MultiGEC Grammar Correction Training Pipeline"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Project dir: ${PROJECT_DIR}"
echo ""

# ── Step 1: Parse raw data ──────────────────────────────────────
echo "[Step 1/6] Parsing MultiGEC data..."
python -m src.data.parse_multigec \
    --data_dir data \
    --output_dir processed_data

# ── Step 2: Data augmentation ───────────────────────────────────
echo ""
echo "[Step 2/6] Augmenting training data..."
python -m src.data.augmentation \
    --input processed_data/multigec_train.jsonl \
    --output processed_data/multigec_train_aug.jsonl \
    --identity_ratio 0.1 \
    --noise_ratio 0.2 \
    --balance_strategy sqrt

# ── Step 3: Prepare verl parquet files ──────────────────────────
echo ""
echo "[Step 3/6] Preparing verl-format Parquet files..."
python -m src.data.prepare_verl_data \
    --input processed_data/multigec_train_aug.jsonl \
    --output_sft processed_data/multigec_train_sft.parquet \
    --output_grpo processed_data/multigec_train_grpo.parquet

python -m src.data.prepare_verl_data \
    --input processed_data/multigec_dev.jsonl \
    --output_grpo processed_data/multigec_dev_grpo.parquet

# ── Step 4: SFT Training ───────────────────────────────────────
if [ "${SKIP_SFT}" != "1" ]; then
    echo ""
    echo "[Step 4/6] SFT Training..."
    deepspeed --num_gpus=2 scripts/train_sft.py \
        --config configs/sft_config.yaml
    SFT_MODEL="outputs/sft/final"
else
    echo ""
    echo "[Step 4/6] Skipping SFT (SKIP_SFT=1)"
    SFT_MODEL=${MODEL_NAME}
fi

# ── Step 5: GRPO Training ──────────────────────────────────────
if [ "${SKIP_GRPO}" != "1" ]; then
    echo ""
    echo "[Step 5/6] GRPO Training..."
    bash scripts/train_grpo.sh \
        "${SFT_MODEL}" \
        processed_data/multigec_train_grpo.parquet \
        processed_data/multigec_dev_grpo.parquet
else
    echo ""
    echo "[Step 5/6] Skipping GRPO (SKIP_GRPO=1)"
fi

# ── Step 6: Evaluation ─────────────────────────────────────────
echo ""
echo "[Step 6/6] Running evaluation on dev set..."

# Evaluate SFT model
if [ "${SKIP_SFT}" != "1" ]; then
    echo "Evaluating SFT model..."
    python scripts/inference.py \
        --model_path outputs/sft/final \
        --input processed_data/multigec_dev.jsonl \
        --output outputs/predictions_sft_dev.jsonl \
        --batch_size 8

    python -m src.eval.evaluate \
        --predictions outputs/predictions_sft_dev.jsonl \
        --references processed_data/multigec_dev.jsonl \
        --output outputs/metrics_sft_dev.json
fi

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "=============================================="
