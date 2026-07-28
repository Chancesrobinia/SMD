#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=10
num_agents=10
algo="rmappo" #"mappo" "ippo"
exp="hetero_graph"
seed_max=1

# =============================================================================
# Heterogeneous Dual-Graph Fusion (L2A + A2A) parameters
#
# Observation layout for original MAPPO simple_spread (N=10, M=10):
#   agent_state   = p_vel(2) + p_pos(2)                        = 4
#   landmark_obs  = entity_pos(10 * 2)                         = 20
#   neighbor_tail = other_pos(9*2) + comm(9*2)                 = 36
#   total obs_dim = 4 + 20 + 36                                = 60
# =============================================================================
agent_state_dim=4
landmark_dim=2
neighbor_dim=4
k_max=8
top_k_filter=6
beta_intrinsic=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb --wandb_name "xxx" --user_name "yuchao" \
    --use_hetero_graph --use_gated_fusion\
    --agent_state_dim ${agent_state_dim} --landmark_dim ${landmark_dim} \
    --neighbor_dim ${neighbor_dim}  --k_max ${k_max} --top_k_filter ${top_k_filter} \
    --beta_intrinsic ${beta_intrinsic}
done