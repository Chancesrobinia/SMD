#!/bin/bash
# =============================================================================
# Train MPE simple_world_comm with adversary-only HeteroGraph-MAPPO.
#
# This keeps the original separated-policy MAPPO training path required by
# simple_world_comm. Adversary actors use the HeteroGraph reasoning module;
# good-agent actors keep the vanilla MAPPO MLP/RNN backbone.
#
# Observation adapter:
#   adversary obs -> ego(8), entities(5 x 4), neighbors(5 x 5)
#   good obs      -> ego(8), entities(5 x 4), neighbors(5 x 5)
#
# NOTE:
#   --share_policy is a store_false flag in this codebase. Passing it disables
#   policy sharing, which is required because adversaries and good agents have
#   different observation dimensions in simple_world_comm.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_world_comm"
num_landmarks=1
num_good_agents=4
num_adversaries=8
num_entities=5
num_forests=2
algo="rmappo"
exp="hetero_graph_adv_only_world_comm"
seed_max=1

agent_state_dim=8
landmark_dim=4
neighbor_dim=5
k_max=5
top_k_filter=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in $(seq ${seed_max});
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
        --env_name ${env} \
        --algorithm_name ${algo} \
        --experiment_name ${exp} \
        --scenario_name ${scenario} \
        --num_good_agents ${num_good_agents} \
        --num_adversaries ${num_adversaries} \
        --num_landmarks ${num_landmarks} \
        --num_entities ${num_entities} \
        --num_forests ${num_forests} \
        --dim_c 4 \
        --seed ${seed} \
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
        --use_hetero_graph \
        --hetero_graph_adversary_only \
        --agent_state_dim ${agent_state_dim} \
        --landmark_dim ${landmark_dim} \
        --neighbor_dim ${neighbor_dim} \
        --k_max ${k_max} \
        --top_k_filter ${top_k_filter}
done
