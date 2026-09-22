#!/bin/sh
# =============================================================================
# Train pure-baseline MAPPO on MPE simple_world_comm
#
# Environment:  simple_world_comm  (2 types of heterogeneous agents)
#   - Adversaries (predators):  4  (all equal, no leader, all silent)
#   - Good agents (prey):       2
#   - Total agents:             6  (auto-computed from good + adv)
#   - Landmarks:                1  (+ 2 food + 2 forests added in scenario)
#
# Key flags:
#   --share_policy        Uses store_false → passing the flag sets it to False
#                         REQUIRED because adversary vs good have different obs dims.
#   --use_valuenorm       Defaults to True; kept explicit for clarity.
#                         Essential for adversarial environments with high variance.
#   --use_centralized_V   Defaults to True; each agent gets centralized value.
#
# This runs vanilla MAPPO (no ESMG / intrinsic reward) since --self_dim and
# --neighbor_dim are not specified.
# =============================================================================

# Auto-cd to onpolicy/scripts so relative paths work regardless of invocation dir
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_world_comm"
num_landmarks=1
num_good_agents=3
num_adversaries=6
algo="rmappo"
exp="rmappo_baseline_world_comm"
seed_max=1

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
        --share_policy
done