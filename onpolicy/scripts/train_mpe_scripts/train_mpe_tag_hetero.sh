#!/bin/sh
# =============================================================================
# Train adversary-only HeteroGraph-MAPPO on MPE simple_tag.
#
# simple_tag observation adapter:
#   adversary obs -> ego(4), landmarks(2 x 2), neighbors(3 x 5)
#   good obs      -> ego(4), landmarks(2 x 2), neighbors(3 x 5)
#
# Each agent has an independent separated policy. Adversary actors use the
# HeteroGraph backbone; good-agent actors use vanilla MAPPO. The critic remains
# the original centralized critic.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_tag"
num_landmarks=2
num_good_agents=3
num_adversaries=6
algo="rmappo"
exp="hetero_graph_adv_only_tag"
seed_max=1

agent_state_dim=4
landmark_dim=2
neighbor_dim=5
k_max=5
top_k_filter=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $(seq "${seed_max}"); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
        --env_name "${env}" \
        --algorithm_name "${algo}" \
        --experiment_name "${exp}" \
        --scenario_name "${scenario}" \
        --num_good_agents "${num_good_agents}" \
        --num_adversaries "${num_adversaries}" \
        --num_landmarks "${num_landmarks}" \
        --seed "${seed}" \
        --n_training_threads 1 \
        --n_rollout_threads 128 \
        --num_mini_batch 1 \
        --episode_length 25 \
        --num_env_steps 10000000 \
        --ppo_epoch 15 \
        --use_ReLU \
        --gain 0.01 \
        --lr 7e-4 \
        --critic_lr 7e-4 \
        --use_valuenorm \
        --use_wandb \
        --user_name "yuchao" \
        --share_policy \
        --use_gated_fusion \
        --use_hetero_graph \
        --hetero_graph_adversary_only \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --k_max "${k_max}" \
        --top_k_filter "${top_k_filter}"
done
