#!/usr/bin/env bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}/onpolicy/scripts"
PYTHON_BIN="${PYTHON_BIN:-/home/lab416/anaconda3/envs/marl_cs/bin/python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" train/train_mpe.py \
  --env_name MPE \
  --algorithm_name rmappo \
  --experiment_name gsd_sparse_smd \
  --scenario_name simple_spread \
  --num_agents 10 \
  --num_landmarks 10 \
  --seed "${SEED:-1}" \
  --n_training_threads 1 \
  --n_rollout_threads 128 \
  --num_mini_batch 1 \
  --episode_length 25 \
  --num_env_steps 20000000 \
  --ppo_epoch 10 \
  --use_ReLU \
  --gain 0.01 \
  --lr 7e-4 \
  --critic_lr 7e-4 \
  --use_wandb \
  --user_name "${USER_NAME:-marl}" \
  --use_hetero_graph \
  --use_gated_fusion \
  --agent_state_dim 4 \
  --landmark_dim 2 \
  --neighbor_dim 4 \
  --k_max 8 \
  --top_k_filter 6 \
  --beta_intrinsic 0 \
  --use_smd \
  --smd_candidate_top_k 3 \
  --smd_edge_top_m 1 \
  --smd_pseudo_top_m 1 \
  --lambda_smd_diff 0.01 \
  --lambda_smd_distill 0.05 \
  --lambda_smd_sparse 0.001 \
  --smd_target_degree 1.0 \
  "$@"
