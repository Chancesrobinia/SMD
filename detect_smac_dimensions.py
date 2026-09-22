#!/usr/bin/env python3
"""Diagnostic: report the true SMAC observation layout for a map.

This is a *diagnostic* tool only. Training no longer needs it: the SMAC
HeteroGraph derives every dimension from the environment's own structured
observation space at runtime (see onpolicy/algorithms/utils/smac_obs_spec.py).

Usage:
    python detect_smac_dimensions.py            # all maps below
    python detect_smac_dimensions.py 3m MMM2    # specific maps
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onpolicy.algorithms.utils.smac_obs_spec import SMACObsSpec
from onpolicy.config import get_config
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from onpolicy.scripts.train.train_smac import parse_args

DEFAULT_MAPS = [
    "3m", "8m", "25m",
    "2m_vs_1z", "3s_vs_3z", "3s_vs_4z", "3s_vs_5z", "5m_vs_6m", "8m_vs_9m",
    "10m_vs_11m", "27m_vs_30m",
    "2s_vs_1sc", "1c3s5z", "3s5z", "3s5z_vs_3s6z", "6h_vs_8z",
    "corridor", "MMM", "MMM2", "2c_vs_64zg",
    "bane_vs_bane", "baneling",
]


def inspect(map_name):
    """Print the exact per-frame layout and verify the fields sum correctly."""
    args = parse_args(['--map_name', map_name], get_config())
    env = StarCraft2Env(args)
    structure = env.get_obs_size()

    frame_obs_dim = structure[0]
    if env.use_stacked_frames:
        frame_obs_dim //= env.stacked_frames

    n_allies, ally_feat_dim = structure[1]
    n_enemies, enemy_feat_dim = structure[2]
    move_feat_dim = structure[3][1]
    own_plus_identity_dim = structure[4][1]

    expected_total = (n_allies * ally_feat_dim
                      + n_enemies * enemy_feat_dim
                      + move_feat_dim
                      + own_plus_identity_dim)
    assert expected_total == frame_obs_dim, (
        '{}: fields sum to {} but frame obs dim is {}'.format(
            map_name, expected_total, frame_obs_dim))

    agent_state_dim = move_feat_dim + own_plus_identity_dim

    print('=' * 64)
    print('map                   : {}'.format(map_name))
    print('total_obs_dim         : {}'.format(structure[0]))
    if env.use_stacked_frames:
        print('  stacked_frames      : {} (frame_obs_dim={})'.format(
            env.stacked_frames, frame_obs_dim))
    print('n_agents              : {}'.format(env.n_agents))
    print('n_allies              : {}'.format(n_allies))
    print('ally_feat_dim         : {}'.format(ally_feat_dim))
    print('n_enemies             : {}'.format(n_enemies))
    print('enemy_feat_dim        : {}'.format(enemy_feat_dim))
    print('move_feat_dim         : {}'.format(move_feat_dim))
    print('own_plus_identity_dim : {}'.format(own_plus_identity_dim))
    print('agent_state_dim       : {}  (= move + own_plus_identity)'.format(
        agent_state_dim))
    print('ctx_dim               : {}  (= move_feat_dim)'.format(move_feat_dim))
    print('field sum check       : {} == {} OK'.format(
        expected_total, frame_obs_dim))

    if not env.use_stacked_frames:
        spec = SMACObsSpec.from_obs_space(structure)
        assert spec.agent_state_dim == agent_state_dim
        assert spec.ctx_dim == move_feat_dim
        print('SMACObsSpec           : consistent')

    env.close()
    return {
        'map_name': map_name,
        'n_agents': env.n_agents,
        'n_allies': n_allies,
        'ally_feat_dim': ally_feat_dim,
        'n_enemies': n_enemies,
        'enemy_feat_dim': enemy_feat_dim,
        'move_feat_dim': move_feat_dim,
        'own_plus_identity_dim': own_plus_identity_dim,
        'agent_state_dim': agent_state_dim,
        'frame_obs_dim': frame_obs_dim,
    }


def main():
    maps = sys.argv[1:] or DEFAULT_MAPS
    results = []
    for map_name in maps:
        try:
            results.append(inspect(map_name))
        except Exception as exc:  # noqa: BLE001 - diagnostic tool
            print('=' * 64)
            print('map {}: FAILED ({})'.format(map_name, exc))

    if len(results) > 1:
        print()
        print('=' * 64)
        print('SUMMARY (all dimensions are read from the environment)')
        print('=' * 64)
        header = '{:<16}{:>7}{:>8}{:>10}{:>9}{:>11}{:>7}{:>13}'.format(
            'map', 'agents', 'allies', 'ally_dim', 'enemies', 'enemy_dim',
            'move', 'agent_state')
        print(header)
        for r in results:
            print('{:<16}{:>7}{:>8}{:>10}{:>9}{:>11}{:>7}{:>13}'.format(
                r['map_name'], r['n_agents'], r['n_allies'],
                r['ally_feat_dim'], r['n_enemies'], r['enemy_feat_dim'],
                r['move_feat_dim'], r['agent_state_dim']))


if __name__ == '__main__':
    main()
