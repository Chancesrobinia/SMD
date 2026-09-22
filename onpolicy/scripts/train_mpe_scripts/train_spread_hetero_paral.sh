#!/bin/sh
# =============================================================================
# Train MPE simple_spread with Parallel 3-Branch HeteroGraph-MAPPO.
#
# HeteroGraphActorBase now uses a fork-join architecture:
#   Macro Peer Branch   : ego -> neighbors
#   Context Branch      : ego -> landmarks
#   Micro Peer Branch   : ego-conditioned KNN/AM filter -> key neighbors
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_spread"
num_landmarks=8
num_agents=8
algo="rmappo"
exp="hetero_graph_parallel_spread"
seed_max=1

agent_state_dim=4
landmark_dim=2
neighbor_dim=4
k_max=6
top_k_filter=3
beta_intrinsic=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $(seq "${seed_max}"); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
        --env_name "${env}" \
        --algorithm_name "${algo}" \
        --experiment_name "${exp}" \
        --scenario_name "${scenario}" \
        --num_agents "${num_agents}" \
        --num_landmarks "${num_landmarks}" \
        --seed "${seed}" \
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
        --wandb_name "xxx" \
        --user_name "yuchao" \
        --use_hetero_graph \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --k_max "${k_max}" \
        --top_k_filter "${top_k_filter}" \
        --beta_intrinsic "${beta_intrinsic}"
done
