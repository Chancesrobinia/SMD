#!/bin/sh
# =============================================================================
# Train parallel AllyGraph + AllEntityGraph on MPE simple_world_comm.
#
# This keeps the separated-policy adversary-only path required by
# simple_world_comm. Adversary actors use the parallel hetero graph backbone;
# good-agent actors keep the vanilla MAPPO actor.
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
dim_c=4
algo="rmappo"
exp="parallel_ally_entity_graph_world_comm"
seed_max=1

agent_state_dim=8
landmark_dim=4
neighbor_dim=5
beta_intrinsic=0

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
        --num_entities "${num_entities}" \
        --num_forests "${num_forests}" \
        --dim_c "${dim_c}" \
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
        --wandb_name "parallel_ally_entity_graph_world_comm" \
        --user_name "yuchao" \
        --share_policy \
        --use_hetero_graph \
        --hetero_graph_adversary_only \
        --use_parallel_ally_graph \
        --use_gated_fusion \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --beta_intrinsic "${beta_intrinsic}"
done
