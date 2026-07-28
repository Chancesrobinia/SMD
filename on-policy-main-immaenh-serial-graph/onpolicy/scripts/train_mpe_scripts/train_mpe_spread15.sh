#!/bin/sh
env="MPE"
scenario="simple_spread" 
num_landmarks=15
num_agents=15
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

# ESMG parameters
# Observation layout for simple_spread (N_agents=10, N_landmarks=10):
#   self_obs  = p_vel(2) + p_pos(2) + entity_pos(10*2) = 24
#   per-neighbor = other_pos(2) + comm(2) = 4, with 9 neighbors => 36
self_dim=34
neighbor_dim=4
k_max=3        # at most 9 neighbors exist
top_k_filter=5
num_graph_layers=2
beta_intrinsic=0.01

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb --wandb_name "xxx" --user_name "yuchao" \
    --self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} \
    --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic}
donepi