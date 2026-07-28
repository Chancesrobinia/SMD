#!/bin/sh
# =============================================================================
# Train GSD-Sparse-SMD on MPE simple_world_comm.
#
# Separated-policy world_comm:
#   adversary actors use HeteroGraph + Sparse Mask Diffusion (SMD)
#   good-agent actors use vanilla MAPPO when --hetero_graph_adversary_only is set.
#
# Agent counts are configurable:
#   NUM_GOOD_AGENTS=3 NUM_ADVERSARIES=6 sh train_mpe_world_SMD.sh
#   NUM_GOOD_AGENTS=4 NUM_ADVERSARIES=8 sh train_mpe_world_SMD.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="${PYTHON_BIN:-/home/lab416/anaconda3/envs/marl_cs/bin/python}"

env="MPE"
scenario="simple_world_comm"
num_landmarks="${NUM_LANDMARKS:-1}"
num_good_agents="${NUM_GOOD_AGENTS:-4}"
num_adversaries="${NUM_ADVERSARIES:-8}"
num_entities="${NUM_ENTITIES:-5}"
num_forests="${NUM_FORESTS:-2}"
dim_c="${DIM_C:-4}"
algo="rmappo"
exp="${EXP_NAME:-gsd_sparse_smd_world_comm}"
seed_max="${SEED_MAX:-1}"

# world_comm ego state for HeteroGraph:
#   p_vel/p_pos(4) + in_forest(num_forests) + prey_forest(num_good_agents)
agent_state_dim=$((4 + num_forests + num_good_agents))
landmark_dim=4
neighbor_dim=5
k_max=5
top_k_filter=3

smd_candidate_top_k="${SMD_CANDIDATE_TOP_K:-5}"
smd_edge_top_m="${SMD_EDGE_TOP_M:-2}"
smd_pseudo_top_m="${SMD_PSEUDO_TOP_M:-1}"
lambda_smd_diff="${LAMBDA_SMD_DIFF:-0.01}"
lambda_smd_distill="${LAMBDA_SMD_DISTILL:-0.05}"
lambda_smd_sparse="${LAMBDA_SMD_SPARSE:-0.001}"
smd_target_degree="${SMD_TARGET_DEGREE:-1.0}"

echo "env=${env}, scenario=${scenario}, algo=${algo}, exp=${exp}, seeds=1..${seed_max}"
echo "world_comm: good=${num_good_agents}, adversary=${num_adversaries}, entities=${num_entities}, forests=${num_forests}"
echo "SMD: candidate_top_k=${smd_candidate_top_k}, edge_top_m=${smd_edge_top_m}, pseudo_top_m=${smd_pseudo_top_m}"

for seed in $(seq "${seed_max}"); do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" train/train_mpe.py \
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
        --gain 0.01 \
        --lr 7e-4 \
        --critic_lr 7e-4 \
        --use_wandb \
        --user_name "yuchao" \
        --share_policy \
        --use_hetero_graph \
        --hetero_graph_adversary_only \
        --use_gated_fusion \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --k_max "${k_max}" \
        --top_k_filter "${top_k_filter}" \
        --smd_candidate_top_k "${smd_candidate_top_k}" \
        --smd_edge_top_m "${smd_edge_top_m}" \
        --smd_pseudo_top_m "${smd_pseudo_top_m}" \
        --lambda_smd_diff "${lambda_smd_diff}" \
        --lambda_smd_distill "${lambda_smd_distill}" \
        --lambda_smd_sparse "${lambda_smd_sparse}" \
        --smd_target_degree "${smd_target_degree}"
done
