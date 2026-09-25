#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if (( $# > 0 )); then
    map="$1"
    shift
else
    map="${MAP_NAME:-}"
fi

if [[ -z "${map}" ]]; then
    echo "Usage: $0 <map_name> [additional train_smac.py arguments]" >&2
    echo "Or set MAP_NAME=<map_name>." >&2
    exit 2
fi

algo_default="rmappo"
ppo_epoch_default="15"
num_mini_batch_default="1"
clip_param_default=""
gain_default=""

# Preserve the tuned settings from the original per-map MAPPO scripts.
# Stacked frames are intentionally not carried over because the classic-SMAC
# SMD observation adapter consumes one frame of entity metadata at a time.
case "${map}" in
    3s5z|3s5z_vs_3s6z|27m_vs_30m|2c_vs_64zg)
        ppo_epoch_default="5"
        ;;
    5m_vs_6m)
        ppo_epoch_default="10"
        clip_param_default="0.05"
        ;;
    8m_vs_9m)
        clip_param_default="0.05"
        ;;
    10m_vs_11m|25m)
        ppo_epoch_default="10"
        ;;
    MMM2)
        ppo_epoch_default="5"
        num_mini_batch_default="2"
        gain_default="1"
        ;;
    3s_vs_4z|6h_vs_8z|corridor)
        algo_default="mappo"
        ppo_epoch_default="5"
        [[ "${map}" == "3s_vs_4z" ]] && ppo_epoch_default="15"
        ;;
    3s_vs_5z)
        algo_default="mappo"
        clip_param_default="0.05"
        ;;
esac

case "${map}" in
    2m_vs_1z|2s_vs_1sc|2c_vs_64zg)
        smd_candidate_top_k_default="1"
        ;;
    3m|3s_vs_3z|3s_vs_4z|3s_vs_5z)
        smd_candidate_top_k_default="2"
        ;;
    *)
        smd_candidate_top_k_default="3"
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

env_name="${ENV_NAME:-StarCraft2}"
algo="${ALGORITHM_NAME:-${algo_default}}"
exp="${EXP_NAME:-smac_${map}_smd}"
seed="${SEED:-1}"
n_training_threads="${TRAINING_THREADS:-1}"
n_rollout_threads="${ROLLOUT_THREADS:-8}"
num_mini_batch="${NUM_MINI_BATCH:-${num_mini_batch_default}}"
episode_length="${EPISODE_LENGTH:-400}"
num_env_steps="${NUM_ENV_STEPS:-10000000}"
ppo_epoch="${PPO_EPOCH:-${ppo_epoch_default}}"
eval_episodes="${EVAL_EPISODES:-32}"
clip_param="${CLIP_PARAM:-${clip_param_default}}"
gain="${GAIN:-${gain_default}}"
smd_candidate_top_k="${SMD_CANDIDATE_TOP_K:-${smd_candidate_top_k_default}}"
smd_edge_top_m="${SMD_EDGE_TOP_M:-1}"
smd_pseudo_top_m="${SMD_PSEUDO_TOP_M:-1}"
lambda_smd_diff="${LAMBDA_SMD_DIFF:-0.01}"
lambda_smd_distill="${LAMBDA_SMD_DISTILL:-0.05}"
lambda_smd_sparse="${LAMBDA_SMD_SPARSE:-0.001}"
smd_target_degree="${SMD_TARGET_DEGREE:-1.0}"

cmd=(
    "${PYTHON_BIN}" train/train_smac.py
    --env_name "${env_name}"
    --algorithm_name "${algo}"
    --experiment_name "${exp}"
    --map_name "${map}"
    --seed "${seed}"
    --n_training_threads "${n_training_threads}"
    --n_rollout_threads "${n_rollout_threads}"
    --num_mini_batch "${num_mini_batch}"
    --episode_length "${episode_length}"
    --num_env_steps "${num_env_steps}"
    --ppo_epoch "${ppo_epoch}"
    --use_value_active_masks
    --use_hetero_graph
    --use_gated_fusion
    --use_smd
    --smd_candidate_top_k "${smd_candidate_top_k}"
    --smd_edge_top_m "${smd_edge_top_m}"
    --smd_pseudo_top_m "${smd_pseudo_top_m}"
    --lambda_smd_diff "${lambda_smd_diff}"
    --lambda_smd_distill "${lambda_smd_distill}"
    --lambda_smd_sparse "${lambda_smd_sparse}"
    --smd_target_degree "${smd_target_degree}"
)

if [[ -n "${clip_param}" ]]; then
    cmd+=(--clip_param "${clip_param}")
fi
if [[ -n "${gain}" ]]; then
    cmd+=(--gain "${gain}")
fi
if [[ "${USE_EVAL:-1}" != "0" ]]; then
    cmd+=(--use_eval --eval_episodes "${eval_episodes}")
fi
# This repository defines --use_wandb as store_false: passing the flag selects
# local TensorBoard logging, while omitting it enables Weights & Biases.
if [[ "${USE_WANDB:-0}" == "0" ]]; then
    cmd+=(--use_wandb)
fi
if [[ "${SMD_DEBUG_SHAPES:-0}" != "0" ]]; then
    cmd+=(--smd_debug_shapes)
fi
if [[ "${RAW_OBS_FUSION:-1}" != "0" ]]; then
    cmd+=(--use_graph_raw_obs_fusion)
fi

cmd+=("$@")

echo "SMAC SMD training: map=${map}, algo=${algo}, seed=${seed}, steps=${num_env_steps}"
echo "SMD: candidate_top_k=${smd_candidate_top_k}, edge_top_m=${smd_edge_top_m}, pseudo_top_m=${smd_pseudo_top_m}"

if [[ "${DRY_RUN:-0}" != "0" ]]; then
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

exec "${cmd[@]}"
