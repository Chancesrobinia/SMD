#!/bin/sh
# =============================================================================
# Render MPE simple_spread — supports HeteroGraph / ESMG / vanilla MAPPO
#
# Set mode="hetero"  for Heterogeneous Dual-Graph models
# Set mode="esmg"    for ESMG-MAPPO models (trained with --self_dim etc.)
# Set mode="vanilla" for vanilla MAPPO models (trained without graph params)
# =============================================================================

env="MPE"
scenario="simple_spread"
num_landmarks=10
num_agents=10
algo="rmappo"
exp="hetero_graph_render"
seed_max=1

# ---- Mode selector: "hetero" | "esmg" | "vanilla" ----
mode="hetero"

# Shared graph parameters (used by both hetero and esmg)
neighbor_dim=4
k_max=6
top_k_filter=3

# HeteroGraph-specific parameters
agent_state_dim=4
landmark_dim=2

# ESMG-specific parameters (only used when mode=esmg)
self_dim=24
num_graph_layers=2

# Ablation switches (must match training config)
use_knn=true
use_am_filter=true

# Model path
model_dir="results/MPE/simple_spread/rmappo/hetero_graph/run6/models"

# ---- Build args string based on mode ----
graph_args=""
if [ "${mode}" = "hetero" ]; then
    graph_args="--use_hetero_graph --agent_state_dim ${agent_state_dim} --num_landmarks ${num_landmarks} --landmark_dim ${landmark_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter}"
    if [ "${use_knn}" = "false" ]; then
        graph_args="${graph_args} --use_knn"
    fi
    if [ "${use_am_filter}" = "false" ]; then
        graph_args="${graph_args} --use_am_filter"
    fi
    echo "Mode: HeteroGraph (use_knn=${use_knn}, use_am_filter=${use_am_filter})"
elif [ "${mode}" = "esmg" ]; then
    graph_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers}"
    if [ "${use_knn}" = "false" ]; then
        graph_args="${graph_args} --use_knn"
    fi
    if [ "${use_am_filter}" = "false" ]; then
        graph_args="${graph_args} --use_am_filter"
    fi
    echo "Mode: ESMG-MAPPO (use_knn=${use_knn}, use_am_filter=${use_am_filter})"
else
    echo "Mode: Vanilla MAPPO"
fi

echo "env is ${env}, scenario is ${scenario}"
echo "model_dir: ${model_dir}"

for seed in $(seq ${seed_max})
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 1 --use_render --use_wandb --use_ReLU --episode_length 25 --render_episodes 5 \
    ${graph_args} \
    --model_dir "${model_dir}"
done
