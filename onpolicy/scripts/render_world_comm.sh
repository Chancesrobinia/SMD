#!/bin/sh
# =============================================================================
# Render MPE simple_world_comm checkpoints to GIF only.
#
# Modes:
#   sh render_world_comm.sh mappo    # pure MAPPO/RMAPPO baseline
#   sh render_world_comm.sh hetero   # adversary HeteroGraph, good-agent MAPPO
#   sh render_world_comm.sh vaggp    # VA-GGP legacy checkpoint
#
# Optional overrides:
#   MODEL_DIR=results/.../runX/models sh render_world_comm.sh hetero
#   SEED=2 RENDER_EPISODES=10 sh render_world_comm.sh mappo
#   PYTHON_BIN=python3 sh render_world_comm.sh hetero
#
# The mappo and hetero modes default to the 4-good / 8-adversary world_comm
# setting used by train_mpe_world_comm_compare_mappo_hetero.sh.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

MODE="${1:-mappo}"

env="MPE"
scenario="simple_world_comm"
seed="${SEED:-1}"
render_episodes="${RENDER_EPISODES:-5}"
episode_length="${EPISODE_LENGTH:-25}"
ifi="${IFI:-0.1}"

num_landmarks=1
num_good_agents=4
num_adversaries=8
num_entities=5
num_forests=2
dim_c=4

hidden_size=64
agent_state_dim=8
landmark_dim=4
neighbor_dim=5
k_max=5
top_k_filter=3

latest_models_dir() {
    base="$1"
    best_run=-1
    best_dir=""
    expected_agents=$((num_good_agents + num_adversaries))

    if [ ! -d "$base" ]; then
        return 1
    fi

    for dir in "$base"/run*/models; do
        [ -d "$dir" ] || continue
        [ -f "$dir/actor_agent0.pt" ] || continue
        actor_count="$(find "$dir" -maxdepth 1 -name 'actor_agent*.pt' | wc -l)"
        [ "$actor_count" -eq "$expected_agents" ] || continue
        run_name="$(basename "$(dirname "$dir")")"
        run_num="${run_name#run}"
        case "$run_num" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ "$run_num" -gt "$best_run" ]; then
            best_run="$run_num"
            best_dir="$dir"
        fi
    done

    [ -n "$best_dir" ] || return 1
    printf '%s\n' "$best_dir"
}

find_first_models_dir() {
    for exp in "$@"; do
        candidate="$(latest_models_dir "results/${env}/${scenario}/${algo}/${exp}" || true)"
        if [ -n "$candidate" ]; then
            selected_exp="$exp"
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

common_flags="--num_entities ${num_entities} --num_forests ${num_forests} --dim_c ${dim_c}"
mode_flags=""
python_bin="${PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    else
        echo "ERROR: neither '${python_bin}' nor python3 is available."
        exit 1
    fi
fi

case "$MODE" in
    mappo|baseline)
        MODE="mappo"
        algo="rmappo"
        exp="render_mappo_world_comm"
        selected_exp=""
        model_dir="${MODEL_DIR:-$(find_first_models_dir mappo_world_comm_baseline mappo_baseline_world_comm || true)}"
        ;;
    hetero|hetero_graph)
        MODE="hetero"
        algo="rmappo"
        exp="render_hetero_graph_world_comm"
        selected_exp=""
        model_dir="${MODEL_DIR:-$(find_first_models_dir hetero_graph_adv_only_world_comm hetero_graph_world_comm_full hetero_graph_world_comm || true)}"
        mode_flags="--use_hetero_graph --hetero_graph_adversary_only --agent_state_dim ${agent_state_dim} --landmark_dim ${landmark_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter}"
        ;;
    vaggp)
        algo="mappo"
        exp="render_vaggp_world_comm"
        hidden_size=128
        num_good_agents=2
        num_adversaries=4
        selected_exp=""
        model_dir="${MODEL_DIR:-$(find_first_models_dir vaggp_world_comm || true)}"
        mode_flags="--use_vaggp"
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: mappo, hetero, or vaggp."
        exit 1
        ;;
esac

if [ -z "$model_dir" ]; then
    echo "ERROR: no checkpoint found for mode '${MODE}'."
    echo "       Set MODEL_DIR=results/MPE/simple_world_comm/<algo>/<exp>/runX/models and rerun."
    exit 1
fi

if [ ! -d "$model_dir" ]; then
    echo "ERROR: model_dir does not exist: ${model_dir}"
    exit 1
fi

if [ ! -f "$model_dir/actor_agent0.pt" ]; then
    echo "ERROR: model_dir does not contain separated-policy checkpoints: ${model_dir}"
    echo "       Expected actor_agent0.pt, actor_agent1.pt, ..."
    exit 1
fi

if [ -n "$selected_exp" ]; then
    echo "Selected checkpoint exp: ${selected_exp}"
fi
echo "============================================================"
echo "  MODE:        ${MODE}"
echo "  algo:        ${algo}"
echo "  render exp:  ${exp}"
echo "  model_dir:   ${model_dir}"
echo "  world_comm:  good=${num_good_agents}, adversary=${num_adversaries}, entities=${num_entities}, forests=${num_forests}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 "${python_bin}" render/render_mpe.py \
    --save_gifs \
    --env_name "${env}" \
    --algorithm_name "${algo}" \
    --experiment_name "${exp}" \
    --scenario_name "${scenario}" \
    --num_good_agents "${num_good_agents}" \
    --num_adversaries "${num_adversaries}" \
    --num_landmarks "${num_landmarks}" \
    --seed "${seed}" \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --use_render \
    --use_ReLU \
    --hidden_size "${hidden_size}" \
    --episode_length "${episode_length}" \
    --render_episodes "${render_episodes}" \
    --ifi "${ifi}" \
    --share_policy \
    --model_dir "${model_dir}" \
    ${common_flags} \
    ${mode_flags}

echo ""
echo "Done. GIFs saved under results/${env}/${scenario}/${algo}/${exp}/"
