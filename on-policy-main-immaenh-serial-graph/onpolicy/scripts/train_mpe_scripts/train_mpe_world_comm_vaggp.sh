#!/bin/bash
# =============================================================================
# Train MPE simple_world_comm — VA-GGP (Variational Autoregressive Guided
# Graph Policy) for adversary agents + standard MAPPO for good agents
#
# Environment:  simple_world_comm  (2 types of heterogeneous agents)
#   - Adversaries (predators):  4  (all equal, no leader, all silent)
#   - Good agents (prey):       2
#   - Total agents:             6  (auto-computed)
#   - Landmarks:                1  (+ 2 food + 2 forests baked into scenario)
#
# Architecture:
#   Adversary agents (0–3):  AdversaryVGBSMLearner + AdversaryAutoregressiveGuider
#   Good agents (4–5):       Standard R_Actor (MLP) MAPPO
#
# Key flags:
#   --share_policy   store_false → passing the flag sets share_policy=False
#   --use_vaggp      enables VA-GGP for adversary agents
#
# USAGE:
#   bash train_mpe_scripts/train_mpe_world_comm_vaggp.sh
#   (works from any directory)
# =============================================================================

# Auto-cd to onpolicy/scripts so relative paths work regardless of invocation dir
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# Activate the correct conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marl2

env="MPE"
scenario="simple_world_comm"
num_landmarks=1
num_good_agents=2
num_adversaries=4
algo="mappo"
exp="vaggp_world_comm"
seed_max=1

echo "============================================================"
echo " env:       ${env}"
echo " scenario:  ${scenario}"
echo " algo:      ${algo}"
echo " exp:       ${exp}"
echo " VA-GGP:    ON (adversary agents 0..$(( num_adversaries - 1 )))"
echo "============================================================"

for seed in $(seq ${seed_max})
do
    echo "seed: ${seed}"
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
        --n_rollout_threads 32 \
        --num_mini_batch 1 \
        --episode_length 25 \
        --num_env_steps 10000000 \
        --ppo_epoch 10 \
        --use_ReLU \
        --lr 7e-4 \
        --critic_lr 7e-4 \
        --hidden_size 128 \
        --entropy_coef 0.01 \
        --share_policy \
        --use_wandb \
        --use_vaggp \
        --vaggp_lambda_rl 1.0 \
        --vaggp_beta_recon 0.1 \
        --vaggp_beta_kl 0.01 \
        --vaggp_delta_clip 1.5 \
        --vaggp_guider_lr 5e-4 \
        --num_forests 2 \
        --dim_c 4 \
        --num_entities 5
done

