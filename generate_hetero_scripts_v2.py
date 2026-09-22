#!/usr/bin/env python3
"""
Generate training scripts with correct dimensions for each SMAC map.
This version calculates approximate agent_state_dim based on map characteristics.
"""

import os

# All SMAC maps with their configurations
# Format: (n_agents, n_enemies, episode_length, estimated_agent_state_dim)
map_configs = {
    # Symmetric maps - simpler units, smaller state
    "3m": (3, 3, 60, 80),
    "8m": (8, 8, 120, 80),
    "25m": (25, 25, 150, 80),

    # Asymmetric maps
    "2m_vs_1z": (2, 1, 150, 90),
    "3s_vs_3z": (3, 3, 150, 100),
    "3s_vs_4z": (3, 4, 200, 100),
    "3s_vs_5z": (3, 5, 250, 120),
    "5m_vs_6m": (5, 6, 70, 120),
    "8m_vs_9m": (8, 9, 120, 140),
    "10m_vs_11m": (10, 11, 150, 160),
    "27m_vs_30m": (27, 30, 180, 200),

    # Mixed unit maps - more complex, larger state
    "2s_vs_1sc": (2, 1, 300, 100),
    "1c3s5z": (9, 9, 150, 180),
    "3s5z": (8, 8, 150, 160),
    "3s5z_vs_3s6z": (8, 9, 170, 180),
    "6h_vs_8z": (6, 8, 150, 150),

    # Challenge maps
    "corridor": (6, 24, 400, 250),
    "MMM": (10, 10, 150, 200),
    "MMM2": (10, 12, 180, 220),
    "2c_vs_64zg": (2, 64, 400, 600),

    # Special maps
    "bane_vs_bane": (24, 24, 200, 200),
    "baneling": (4, 20, 200, 250),
}

script_template = """#!/bin/sh
env="StarCraft2"
map="{map_name}"
algo="mappo"
exp="hetero_no_stack"
seed_max=1

echo "env is ${{env}}, map is ${{map}}, algo is ${{algo}}, exp is ${{exp}}, max seed is ${{seed_max}}"
for seed in `seq ${{seed_max}}`;
do
    echo "seed is ${{seed}}:"
    CUDA_VISIBLE_DEVICES=0 python ../../train/train_smac.py --env_name ${{env}} --algorithm_name ${{algo}} --experiment_name ${{exp}} \\
    --map_name ${{map}} --seed ${{seed}} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length {episode_length} \\
    --num_env_steps 10000000 --ppo_epoch 15 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 \\
    --use_hetero_graph --hidden_size 64 --neighbor_dim 8 --agent_state_dim {agent_state_dim} \\
    --landmark_dim 4 --num_agents {n_agents} --n_allies {n_allies} --n_enemies {n_enemies} \\
    --ally_feat_dim 8 --enemy_feat_dim 8
done
"""

def generate_script(map_name):
    """Generate training script for a specific map."""
    if map_name not in map_configs:
        print(f"Warning: No config found for map {map_name}, using defaults")
        n_agents, n_enemies, episode_length, agent_state_dim = 8, 8, 150, 120
    else:
        n_agents, n_enemies, episode_length, agent_state_dim = map_configs[map_name]

    n_allies = max(n_agents - 1, 0)

    script_content = script_template.format(
        map_name=map_name,
        n_agents=n_agents,
        n_allies=n_allies,
        n_enemies=n_enemies,
        episode_length=episode_length,
        agent_state_dim=agent_state_dim
    )

    return script_content

def main():
    output_dir = "onpolicy/scripts/train_smac_hetero_scripts"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating training scripts for {len(map_configs)} SMAC maps...")
    print("Note: Using estimated agent_state_dim values. May need adjustment based on actual observations.")

    for map_name in map_configs.keys():
        script_content = generate_script(map_name)
        script_path = os.path.join(output_dir, f"train_smac_{map_name}.sh")

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)
        n_agents, n_enemies, _, agent_state_dim = map_configs[map_name]
        print(f"Created: {script_path:60s} | agents:{n_agents:3d} enemies:{n_enemies:3d} state_dim:{agent_state_dim:3d}")

    print(f"\nSuccessfully generated {len(map_configs)} training scripts in {output_dir}/")
    print("\nIMPORTANT: These scripts use estimated agent_state_dim values.")
    print("If you get dimension mismatch errors, you may need to adjust agent_state_dim for that specific map.")
    print("\nTo run a script:")
    print(f"  cd {output_dir}")
    print(f"  ./train_smac_MMM2.sh")

if __name__ == "__main__":
    main()
