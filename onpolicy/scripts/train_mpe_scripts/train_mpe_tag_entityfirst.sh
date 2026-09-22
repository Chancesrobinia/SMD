#!/bin/sh
# =============================================================================
# Train Entity-First HeteroGraph-MAPPO on MPE simple_tag.
#
# Entity-first path:
#   Hop1: enemy agents + landmarks attention
#   Hop2: ally-agent attention conditioned on Hop1 context
#   Hop3: filtered enemy agents + landmarks attention conditioned on Hop2 intent
#   Fusion: ally intent + refined target context
#
# This keeps the separated-policy adversary-only path used by simple_tag.
# Adversary actors use the entity-first graph backbone; good-agent actors keep
# the vanilla MAPPO actor.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
export PYTHONPATH="$SCRIPT_DIR/../..:${PYTHONPATH:-}"

env="MPE"
scenario="simple_tag"
num_landmarks=2
num_good_agents=4
num_adversaries=8
algo="rmappo"
exp="hetero_graph_entityfirst_tag"
seed_max=1

agent_state_dim=4
landmark_dim=2
neighbor_dim=5
k_max=5
top_k_filter=3
beta_intrinsic=0
use_gated_fusion=1

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
        --wandb_name "hetero_graph_entityfirst_tag" \
        --user_name "yuchao" \
        --share_policy \
        --use_hetero_graph \
        --hetero_graph_adversary_only \
        --use_entity_enemy_first_graph \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --k_max "${k_max}" \
        --top_k_filter "${top_k_filter}" \
        --beta_intrinsic "${beta_intrinsic}" \
        $(if [ "${use_gated_fusion}" = "1" ]; then echo "--use_gated_fusion"; fi)
done
