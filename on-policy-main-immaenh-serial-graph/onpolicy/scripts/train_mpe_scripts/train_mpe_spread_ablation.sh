#!/bin/bash
# Standard ablation suite for MPE simple_spread (serial-graph).
# Usage: bash train_mpe_spread_ablation.sh

set -euo pipefail

env="MPE"
scenario="simple_spread"
num_agents=8
num_landmarks=8
algo="rmappo"
exp_prefix="ablation_spread"
seed_max=1

# Common training hyperparameters
n_rollout_threads=128
episode_length=25
num_env_steps=20000000
ppo_epoch=10
num_mini_batch=1
lr=7e-4
critic_lr=7e-4

# ESMG / HeteroGraph defaults
self_dim=4
neighbor_dim=4
k_max=6
top_k_filter=3
num_graph_layers=2
recon_loss_coef=0.01
beta_intrinsic=0.01

# HeteroGraph split defaults (simple_spread layout)
agent_state_dim=4
landmark_dim=2

run_one() {
  local exp_name="$1"
  local extra_args="$2"

  echo "[RUN] ${exp_name} :: ${extra_args}"
  for seed in $(seq ${seed_max}); do
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py \
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp_name} \
      --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
      --n_training_threads 1 --n_rollout_threads ${n_rollout_threads} --num_mini_batch ${num_mini_batch} \
      --episode_length ${episode_length} --num_env_steps ${num_env_steps} --ppo_epoch ${ppo_epoch} \
      --use_ReLU --gain 0.01 --lr ${lr} --critic_lr ${critic_lr} --use_wandb \
      ${extra_args}
  done
}

# 0) Baseline MAPPO (MLP) - disable graph/ESMG/simhash
run_one "${exp_prefix}_mlp" "--use_simhash_exploration"

# 1) ESMG full (self/neighbor dims + KNN + AM filter + recon + simhash)
run_one "${exp_prefix}_esmg_full" "--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} \
  --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} \
  --recon_loss_coef ${recon_loss_coef} --beta_intrinsic ${beta_intrinsic}"

# 2) ESMG without AM filter + recon
run_one "${exp_prefix}_esmg_no_am" "--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} \
  --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} \
  --use_am_filter --recon_loss_coef 0.0 --beta_intrinsic ${beta_intrinsic}"

# 3) ESMG without KNN (process all neighbors)
run_one "${exp_prefix}_esmg_no_knn" "--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} \
  --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} \
  --use_knn --beta_intrinsic ${beta_intrinsic}"

# 4) ESMG without SimHash intrinsic reward
run_one "${exp_prefix}_esmg_no_simhash" "--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} \
  --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} \
  --recon_loss_coef ${recon_loss_coef} --use_simhash_exploration"

# 5) HeteroGraph (L2A + A2A)
run_one "${exp_prefix}_hetero_graph" "--use_hetero_graph \
  --agent_state_dim ${agent_state_dim} --landmark_dim ${landmark_dim} \
  --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} \
  --beta_intrinsic ${beta_intrinsic}"

# 6) HeteroGraph without SimHash
run_one "${exp_prefix}_hetero_no_simhash" "--use_hetero_graph \
  --agent_state_dim ${agent_state_dim} --landmark_dim ${landmark_dim} \
  --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} \
  --use_simhash_exploration"

