#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
env="StarCraft2"
map="3m"
algo="rmappo"
exp="${EXP_NAME:-smac_3m_smd_smoke}"
seed="${SEED:-1}"

"$PYTHON_BIN" train/train_smac.py \
    --env_name "$env" \
    --algorithm_name "$algo" \
    --experiment_name "$exp" \
    --map_name "$map" \
    --seed "$seed" \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --num_mini_batch 1 \
    --episode_length 60 \
    --num_env_steps "${NUM_ENV_STEPS:-5000}" \
    --ppo_epoch 2 \
    --use_eval \
    --eval_episodes 1 \
    --use_hetero_graph \
    --use_gated_fusion \
    --use_smd \
    --smd_candidate_top_k 2 \
    --smd_edge_top_m 1 \
    --smd_pseudo_top_m 1 \
    --lambda_smd_diff 0.01 \
    --lambda_smd_distill 0.05 \
    --lambda_smd_sparse 0.001 \
    --smd_target_degree 1.0 \
    --smd_debug_shapes \
    --use_wandb
