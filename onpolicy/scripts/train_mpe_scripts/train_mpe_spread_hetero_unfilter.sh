#!/bin/sh
# =============================================================================
# Train serial HeteroGraph on MPE simple_spread without Hop-3 filtering.
#
# Important:
#   In onpolicy/config.py, --use_knn and --use_am_filter are store_false flags.
#   Passing them disables KNN and disables AM/top-k filtering.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_spread"
num_landmarks=8
num_agents=8
algo="rmappo"
exp="hetero_graph_serial_unfilter_spread"
seed_max=1

agent_state_dim=4
landmark_dim=2
neighbor_dim=4
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
        --wandb_name "hetero_graph_serial_unfilter_spread" \
        --user_name "yuchao" \
        --use_hetero_graph \
        --use_gated_fusion \
        --use_knn \
        --use_am_filter \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --beta_intrinsic "${beta_intrinsic}"
done
