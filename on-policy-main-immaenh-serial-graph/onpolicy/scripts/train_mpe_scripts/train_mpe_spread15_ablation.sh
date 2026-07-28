#!/bin/sh
# =============================================================================
# Ablation Study Training Script for ESMG-MAPPO on MPE simple_spread (N=10)
#
# Usage: Set the ABLATION variable to one of:
#   full        - Full ESMG (KNN + AM Filter + 2-layer Multiplex + SimHash)
#   no_knn      - Disable KNN truncation (process all neighbors)
#   knn_k1      - KNN with k=1 (only the single nearest neighbor)
#   no_am       - Disable AM filter (no value-guided pruning, no recon loss)
#   single_layer - Single-layer graph (esmg_layers=1)
#   no_simhash  - Disable SimHash intrinsic exploration reward
#   vanilla     - Vanilla MAPPO baseline (no ESMG at all)
# =============================================================================

env="MPE"
scenario="simple_spread"
num_landmarks=10
num_agents=10
algo="rmappo"
seed_max=1

# ======== SELECT ABLATION HERE ========
ABLATION="${1:-full}"
# =======================================

# Common ESMG parameters
# Observation layout for simple_spread (N_agents=10, N_landmarks=10):
#   self_obs  = p_vel(2) + p_pos(2) + entity_pos(10*2) = 24
#   per-neighbor = other_pos(2) + comm(2) = 4, with 9 neighbors => 36
self_dim=24
neighbor_dim=4
k_max=3
top_k_filter=5
num_graph_layers=2
beta_intrinsic=0.01

case "${ABLATION}" in
    full)
        exp="ablation_full"
        esmg_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic}"
        echo "=== ABLATION: Full ESMG ==="
        ;;
    no_knn)
        exp="ablation_no_knn"
        # --use_knn flag presence sets use_knn=False (action='store_false')
        esmg_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic} --use_knn"
        echo "=== ABLATION: No KNN Truncation ==="
        ;;
    knn_k1)
        exp="ablation_knn_k1"
        # KNN keeps only the single nearest neighbor; top_k_filter clamped to 1 accordingly
        esmg_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max 1 --top_k_filter 1 --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic}"
        echo "=== ABLATION: KNN k=1 (single nearest neighbor) ==="
        ;;
    no_am)
        exp="ablation_no_am"
        # --use_am_filter flag presence sets use_am_filter=False
        esmg_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic} --use_am_filter"
        echo "=== ABLATION: No AM Filter ==="
        ;;
    single_layer)
        exp="ablation_single_layer"
        esmg_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic} --esmg_layers 1"
        echo "=== ABLATION: Single-Layer Graph ==="
        ;;
    no_simhash)
        exp="ablation_no_simhash"
        # --use_simhash_exploration flag presence sets it to False
        esmg_args="--self_dim ${self_dim} --neighbor_dim ${neighbor_dim} --k_max ${k_max} --top_k_filter ${top_k_filter} --num_graph_layers ${num_graph_layers} --beta_intrinsic ${beta_intrinsic} --use_simhash_exploration"
        echo "=== ABLATION: No SimHash Exploration ==="
        ;;
    vanilla)
        exp="ablation_vanilla"
        # No ESMG args → self_dim/neighbor_dim remain None → MLPBase is used
        esmg_args=""
        echo "=== ABLATION: Vanilla MAPPO (no ESMG) ==="
        ;;
    *)
        echo "Unknown ablation: ${ABLATION}"
        echo "Usage: $0 {full|no_knn|knn_k1|no_am|single_layer|no_simhash|vanilla}"
        exit 1
        ;;
esac

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

for seed in $(seq ${seed_max});
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --use_wandb --wandb_name "xxx" --user_name "yuchao " \
    ${esmg_args}
done
