#!/bin/sh
# =============================================================================
# Train vanilla MAPPO baseline on MPE simple_spread
#
# Without --self_dim / --neighbor_dim:
#   - observation() uses original MAPPO layout: [other_pos + comm]
#   - R_Actor falls back to MLPBase
#   - latent_beliefs=None so intrinsic reward = 0
#
# Result is DIRECTLY comparable to ESMG-MAPPO on the same reward function.
# =============================================================================

env="MPE"
scenario="simple_spread"
num_landmarks=15
num_agents=15
algo="rmappo"
exp="mappo_baseline"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb --user_name "yuchao"
done
