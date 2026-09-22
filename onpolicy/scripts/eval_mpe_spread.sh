#!/bin/sh
# =============================================================================
# Evaluate ESMG-MAPPO model on MPE simple_spread
#
# This produces pure-environment-reward numbers that are DIRECTLY comparable
# to vanilla MAPPO evaluation results on the same env config.
#
# Key points:
#   - No intrinsic reward involved (only env reward)
#   - Uses deterministic actions
#   - Same env config (num_agents, num_landmarks, episode_length) as training
#   - ESMG params must match training to load actor.pt correctly
# =============================================================================

env="MPE"
scenario="simple_spread"
num_landmarks=10
num_agents=10
algo="rmappo"
exp="check"

# ESMG parameters (must match training config exactly)
self_dim=24
neighbor_dim=4
k_max=3
top_k_filter=5
num_graph_layers=2

# Evaluation settings
eval_episodes=100
seed=1

# Model path — change this to your run
model_dir="results/MPE/simple_spread/rmappo/check/run13/models"

echo "=========================================="
echo " Evaluating ESMG-MAPPO on ${scenario}"
echo " Model: ${model_dir}"
echo " Episodes: ${eval_episodes}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python eval/eval_mpe_spread.py \
    --algorithm_name ${algo} \
    --scenario_name ${scenario} \
    --env_name ${env} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --episode_length 25 \
    --seed ${seed} \
    --use_wandb --use_ReLU \
    --self_dim ${self_dim} \
    --neighbor_dim ${neighbor_dim} \
    --k_max ${k_max} \
    --top_k_filter ${top_k_filter} \
    --num_graph_layers ${num_graph_layers} \
    --eval_episodes ${eval_episodes} \
    --model_dir "${model_dir}"
