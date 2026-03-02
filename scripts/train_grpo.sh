#!/bin/bash
# GRPO Training Script for MultiGEC Grammar Correction
# Target hardware: 2x NVIDIA A30 (24GB each)
# Model: Qwen2.5-3B-Instruct (or SFT checkpoint)
#
# Usage:
#   bash scripts/train_grpo.sh [MODEL_PATH] [TRAIN_PARQUET] [VAL_PARQUET]
#
# Example:
#   bash scripts/train_grpo.sh Qwen/Qwen2.5-3B-Instruct processed_data/multigec_train.parquet processed_data/multigec_dev.parquet
#   bash scripts/train_grpo.sh outputs/sft/final processed_data/multigec_train.parquet processed_data/multigec_dev.parquet

set -x

MODEL_PATH=${1:-"Qwen/Qwen3-4B"}
TRAIN_FILE=${2:-"processed_data/multigec_train.parquet"}
VAL_FILE=${3:-"processed_data/multigec_dev.parquet"}

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=${PROJECT_DIR}/src/rewards/grammar_reward.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='multigec-grpo' \
    trainer.experiment_name='qwen2.5-3b-grammar-grpo' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 "$@"
