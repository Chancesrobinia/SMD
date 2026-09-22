#!/bin/sh
# =============================================================================
# Train pure HAPPO on MPE simple_world_comm.
#
# simple_world_comm is heterogeneous: adversaries and good agents have different
# observation dimensions, so this script disables shared policy. Parameters match
# the current world_comm hetero/compare scripts, but no graph or VA-GGP modules
# are enabled.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_world_comm"
num_landmarks=1
num_good_agents=3
num_adversaries=6
num_entities=5
num_forests=2
dim_c=4
algo="happo"
exp="happo_world_comm"
seed_max=1

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
        --user_name "yuchao" \
        --share_policy
done
