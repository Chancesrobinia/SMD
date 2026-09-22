#!/bin/sh
# =============================================================================
# Render a trained MAPPO model on MPE simple_world_comm
#
# IMPORTANT: Set --model_dir to the checkpoint directory containing
#            actor_agent0.pt ... actor_agent5.pt, etc.
# =============================================================================
env="MPE"
scenario="simple_world_comm"
num_landmarks=1
num_good_agents=2
num_adversaries=4
algo="rmappo"
exp="mappo_baseline_world_comm"
seed=1
# >>>>>>>  CHANGE THIS to your actual checkpoint directory  <<<<<<<
model_dir="../../scripts/results/MPE/simple_world_comm/rmappo/mappo_baseline_world_comm/run1/models"
echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}"
echo "Loading model from: ${model_dir}"
CUDA_VISIBLE_DEVICES=0 python ../render/render_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_good_agents ${num_good_agents} \
    --num_adversaries ${num_adversaries} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --episode_length 25 \
    --use_ReLU \
    --gain 0.01 \
    --use_valuenorm \
    --use_render \
    --render_episodes 5 \
    --save_gifs \
    --ifi 0.1 \
    --model_dir ${model_dir} \
    --share_policy
