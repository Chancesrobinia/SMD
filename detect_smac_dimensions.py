#!/usr/bin/env python3
"""
Detect actual SMAC observation dimensions for heterograph configuration.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env

def detect_dimensions(map_name, use_stacked_frames=True, stacked_frames=4):
    """Detect observation dimensions for a specific SMAC map."""

    # Create minimal args for environment
    args = argparse.Namespace(
        map_name=map_name,
        add_move_state=False,
        add_local_obs=False,
        add_distance_state=False,
        add_enemy_action_state=False,
        add_agent_id=False,
        add_visible_state=False,
        add_xy_state=False,
        use_state_agent=True,
        use_mustalive=True,
        add_center_xy=True,
        stacked_frames=stacked_frames,
        use_stacked_frames=use_stacked_frames
    )

    try:
        env = StarCraft2Env(args)
        obs_size_info = env.get_obs_size()

        # obs_size_info format: [total_obs, [n_allies, ally_feat], [n_enemies, enemy_feat], [1, move_feat], [1, own_feat]]
        total_obs = obs_size_info[0]
        n_allies, ally_feat_dim = obs_size_info[1]
        n_enemies, enemy_feat_dim = obs_size_info[2]
        _, move_feat_dim = obs_size_info[3]
        _, own_feat_dim = obs_size_info[4]

        # Calculate agent_state_dim
        agent_state_dim = total_obs - (n_allies * ally_feat_dim) - (n_enemies * enemy_feat_dim)

        print(f"\n{'='*60}")
        print(f"Map: {map_name}")
        print(f"{'='*60}")
        print(f"Total observation size: {total_obs}")
        print(f"Number of agents: {env.n_agents}")
        print(f"Allies: {n_allies}, Ally feature dim: {ally_feat_dim}")
        print(f"Enemies: {n_enemies}, Enemy feature dim: {enemy_feat_dim}")
        print(f"Move features: {move_feat_dim}")
        print(f"Own features: {own_feat_dim}")
        print(f"Agent state dim: {agent_state_dim}")
        print(f"\nRecommended parameters:")
        print(f"  --num_agents {env.n_agents}")
        print(f"  --n_allies {n_allies}")
        print(f"  --n_enemies {n_enemies}")
        print(f"  --ally_feat_dim {ally_feat_dim}")
        print(f"  --enemy_feat_dim {enemy_feat_dim}")
        print(f"  --agent_state_dim {agent_state_dim}")
        print(f"  --landmark_dim {min(move_feat_dim, 4)}")

        env.close()
        return {
            'map_name': map_name,
            'num_agents': env.n_agents,
            'n_allies': n_allies,
            'n_enemies': n_enemies,
            'ally_feat_dim': ally_feat_dim,
            'enemy_feat_dim': enemy_feat_dim,
            'agent_state_dim': agent_state_dim,
            'move_feat_dim': move_feat_dim,
            'own_feat_dim': own_feat_dim,
            'total_obs': total_obs
        }
    except Exception as e:
        print(f"\nError detecting dimensions for {map_name}: {e}")
        return None

def main():
    maps = [
        "3m", "8m", "25m",
        "2m_vs_1z", "3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "5m_vs_6m", "8m_vs_9m", "10m_vs_11m", "27m_vs_30m",
        "2s_vs_1sc", "1c3s5z", "3s5z", "3s5z_vs_3s6z", "6h_vs_8z",
        "corridor", "MMM", "MMM2", "2c_vs_64zg",
        "bane_vs_bane", "baneling"
    ]

    # Detect dimensions for a single map or all maps
    if len(sys.argv) > 1:
        map_name = sys.argv[1]
        detect_dimensions(map_name)
    else:
        print("Detecting dimensions for all SMAC maps...")
        results = []
        for map_name in maps:
            result = detect_dimensions(map_name)
            if result:
                results.append(result)

        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"{r['map_name']:20s} | agents:{r['num_agents']:3d} | allies:{r['n_allies']:3d} | enemies:{r['n_enemies']:3d} | "
                  f"ally_feat:{r['ally_feat_dim']:2d} | enemy_feat:{r['enemy_feat_dim']:2d} | agent_state:{r['agent_state_dim']:4d}")

if __name__ == "__main__":
    main()
