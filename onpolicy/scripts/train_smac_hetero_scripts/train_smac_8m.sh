#!/bin/sh
env="StarCraft2"
map="8m"
algo="mappo"
exp="hetero_no_stack"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 120 \
    --num_env_steps 10000000 --ppo_epoch 15 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 \
    --use_hetero_graph --hidden_size 64 --neighbor_dim 8 --agent_state_dim 80 \
    --landmark_dim 4 --num_agents 8 --n_allies 7 --n_enemies 8 \
    --ally_feat_dim 8 --enemy_feat_dim 8
done
