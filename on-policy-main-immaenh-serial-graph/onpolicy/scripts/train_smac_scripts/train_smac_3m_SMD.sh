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
exp="${EXP_NAME:-smac_3m_smd}"
seed="${SEED:-1}"

"$PYTHON_BIN" train/train_smac.py \
    --env_name "$env" \
    --algorithm_name "$algo" \
    --experiment_name "$exp" \
    --map_name "$map" \
    --seed "$seed" \
    --n_training_threads 1 \
    --n_rollout_threads "${ROLLOUT_THREADS:-8}" \
    --num_mini_batch 1 \
    --episode_length 400 \
    --num_env_steps "${NUM_ENV_STEPS:-10000000}" \
    --ppo_epoch 5 \
    --use_eval \
    --eval_episodes 32 \
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
    --use_wandb
