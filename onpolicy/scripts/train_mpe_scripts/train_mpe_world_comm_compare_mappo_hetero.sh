#!/bin/sh
# =============================================================================
# Paired comparison on MPE simple_world_comm:
#   1) mappo_world_comm_baseline      flat observation -> MLP/RNN actor
#   2) hetero_graph_adv_only_world_comm   adversary HeteroGraph, good-agent MAPPO
#
# Optional hetero ablations can be enabled with:
#   RUN_HETERO_ABLATIONS=1 sh train_mpe_world_comm_compare_mappo_hetero.sh
#
# All cases use the same world_comm setting, seeds, PPO hyperparameters,
# centralized critic, and separated-policy runner. The intended difference is
# the adversary local actor representation backbone, plus optional graph ablation flags.
#
# NOTE:
#   --share_policy is a store_false flag in this codebase. Passing it disables
#   policy sharing, which is required because adversaries and good agents have
#   different observation dimensions in simple_world_comm.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

env="MPE"
scenario="simple_world_comm"
algo="rmappo"
seed_max="${SEED_MAX:-1}"
run_hetero_ablations="${RUN_HETERO_ABLATIONS:-0}"

# simple_world_comm currently hardcodes 2 food and only checks forest[0]/forest[1].
num_landmarks=1
num_good_agents=4
num_adversaries=8
num_entities=5
num_forests=2
dim_c=4

n_training_threads=1
n_rollout_threads=128
num_mini_batch=1
episode_length=25
num_env_steps=10000000
ppo_epoch=15
lr=7e-4
critic_lr=7e-4
gain=0.01

# world_comm hetero adapter:
#   adversary obs -> ego(8), entities(5 x 4), neighbors((N-1) x 5)
#   good obs      -> ego(8), entities(5 x 4), neighbors((N-1) x 5)
agent_state_dim=8
landmark_dim=4
neighbor_dim=5
k_max=5
top_k_filter=3

run_common() {
    common_seed="$1"
    common_exp="$2"
    shift 2

    CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
        --env_name "${env}" \
        --algorithm_name "${algo}" \
        --experiment_name "${common_exp}" \
        --scenario_name "${scenario}" \
        --num_good_agents "${num_good_agents}" \
        --num_adversaries "${num_adversaries}" \
        --num_landmarks "${num_landmarks}" \
        --num_entities "${num_entities}" \
        --num_forests "${num_forests}" \
        --dim_c "${dim_c}" \
        --seed "${common_seed}" \
        --n_training_threads "${n_training_threads}" \
        --n_rollout_threads "${n_rollout_threads}" \
        --num_mini_batch "${num_mini_batch}" \
        --episode_length "${episode_length}" \
        --num_env_steps "${num_env_steps}" \
        --ppo_epoch "${ppo_epoch}" \
        --use_ReLU \
        --gain "${gain}" \
        --lr "${lr}" \
        --critic_lr "${critic_lr}" \
        --use_valuenorm \
        --use_wandb \
        --user_name "yuchao" \
        --share_policy \
        "$@"
}

run_case() {
    case_seed="$1"
    case_name="$2"
    shift 2

    echo "============================================================"
    echo "seed ${case_seed}: ${case_name}"
    echo "============================================================"
    run_common "${case_seed}" "${case_name}" "$@"
}

echo "hetero comparison: scenario=${scenario}, algo=${algo}, seeds=1..${seed_max}"
echo "world_comm: good=${num_good_agents}, adversary=${num_adversaries}, entities=${num_entities}, forests=${num_forests}"

for seed in $(seq "${seed_max}"); do
    run_case "${seed}" "mappo_world_comm_baseline"

    run_case "${seed}" "hetero_graph_adv_only_world_comm" \
        --use_hetero_graph \
        --hetero_graph_adversary_only \
        --agent_state_dim "${agent_state_dim}" \
        --landmark_dim "${landmark_dim}" \
        --neighbor_dim "${neighbor_dim}" \
        --k_max "${k_max}" \
        --top_k_filter "${top_k_filter}"

    if [ "${run_hetero_ablations}" = "1" ]; then
        # In config.py these two flags are store_false: passing them disables the component.
        run_case "${seed}" "hetero_graph_world_comm_no_knn" \
            --use_hetero_graph \
            --hetero_graph_adversary_only \
            --agent_state_dim "${agent_state_dim}" \
            --landmark_dim "${landmark_dim}" \
            --neighbor_dim "${neighbor_dim}" \
            --k_max "${k_max}" \
            --top_k_filter "${top_k_filter}" \
            --use_knn

        run_case "${seed}" "hetero_graph_world_comm_no_am" \
            --use_hetero_graph \
            --hetero_graph_adversary_only \
            --agent_state_dim "${agent_state_dim}" \
            --landmark_dim "${landmark_dim}" \
            --neighbor_dim "${neighbor_dim}" \
            --k_max "${k_max}" \
            --top_k_filter "${top_k_filter}" \
            --use_am_filter
    fi
done
