#!/usr/bin/env python3
"""
Generate training scripts for all SMAC maps with heterograph support.
"""

import os

# All SMAC maps
maps = [
    "3m", "8m", "25m",
    "2m_vs_1z", "3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "5m_vs_6m", "8m_vs_9m", "10m_vs_11m", "27m_vs_30m",
    "2s_vs_1sc", "1c3s5z", "3s5z", "3s5z_vs_3s6z", "6h_vs_8z",
    "corridor", "MMM", "MMM2", "2c_vs_64zg",
    "bane_vs_bane", "baneling"
]

# Map-specific configurations (n_agents, n_enemies for heterograph dimensions)
# Format: map_name -> (n_agents, n_enemies, episode_length)
map_configs = {
    "3m": (3, 3, 60),
    "8m": (8, 8, 120),
    "25m": (25, 25, 150),
    "2m_vs_1z": (2, 1, 150),
    "3s_vs_3z": (3, 3, 150),
    "3s_vs_4z": (3, 4, 200),
    "3s_vs_5z": (3, 5, 250),
    "5m_vs_6m": (5, 6, 70),
    "8m_vs_9m": (8, 9, 120),
    "10m_vs_11m": (10, 11, 150),
    "27m_vs_30m": (27, 30, 180),
    "2s_vs_1sc": (2, 1, 300),
    "1c3s5z": (9, 9, 150),
    "3s5z": (8, 8, 150),
    "3s5z_vs_3s6z": (8, 9, 170),
    "6h_vs_8z": (6, 8, 150),
    "corridor": (6, 24, 400),
    "MMM": (10, 10, 150),
    "MMM2": (10, 12, 180),
    "2c_vs_64zg": (2, 64, 400),
    "bane_vs_bane": (24, 24, 200),
    "baneling": (4, 20, 200),
}

# Default hetero graph parameters
hetero_params = {
    "use_hetero_graph": True,
    "hidden_size": 64,
    "neighbor_dim": 8,  # SMAC ally/enemy feature dimension (adjust based on actual observation)
    "agent_state_dim": 12,  # own_feats + move_feats dimension
    "landmark_dim": 4,  # context dimension (move_feats)
}

script_template = """#!/bin/sh
env="StarCraft2"
map="{map_name}"
algo="mappo"
exp="hetero_check"
seed_max=1

echo "env is ${{env}}, map is ${{map}}, algo is ${{algo}}, exp is ${{exp}}, max seed is ${{seed_max}}"
for seed in `seq ${{seed_max}}`;
do
    echo "seed is ${{seed}}:"
    CUDA_VISIBLE_DEVICES=0 python ../../train/train_smac.py --env_name ${{env}} --algorithm_name ${{algo}} --experiment_name ${{exp}} \\
    --map_name ${{map}} --seed ${{seed}} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length {episode_length} \\
    --num_env_steps 10000000 --ppo_epoch 15 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 \\
    --use_hetero_graph --hidden_size {hidden_size} --neighbor_dim {neighbor_dim} --agent_state_dim {agent_state_dim} \\
    --landmark_dim {landmark_dim} --num_agents {n_agents} --n_allies {n_allies} --n_enemies {n_enemies} \\
    --ally_feat_dim {neighbor_dim} --enemy_feat_dim {neighbor_dim}
done
"""

def generate_script(map_name):
    """Generate training script for a specific map."""
    if map_name not in map_configs:
        print(f"Warning: No config found for map {map_name}, using defaults")
        n_agents, n_enemies, episode_length = 8, 8, 150
    else:
        n_agents, n_enemies, episode_length = map_configs[map_name]

    n_allies = max(n_agents - 1, 0)

    script_content = script_template.format(
        map_name=map_name,
        n_agents=n_agents,
        n_allies=n_allies,
        n_enemies=n_enemies,
        episode_length=episode_length,
        hidden_size=hetero_params["hidden_size"],
        neighbor_dim=hetero_params["neighbor_dim"],
        agent_state_dim=hetero_params["agent_state_dim"],
        landmark_dim=hetero_params["landmark_dim"]
    )

    return script_content

def main():
    output_dir = "onpolicy/scripts/train_smac_hetero_scripts"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating training scripts for {len(maps)} SMAC maps...")

    for map_name in maps:
        script_content = generate_script(map_name)
        script_path = os.path.join(output_dir, f"train_smac_{map_name}.sh")

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)
        print(f"Created: {script_path}")

    print(f"\nSuccessfully generated {len(maps)} training scripts in {output_dir}/")
    print("\nTo run a script, use:")
    print(f"  cd {output_dir}")
    print(f"  ./train_smac_3s_vs_5z.sh")

if __name__ == "__main__":
    main()
