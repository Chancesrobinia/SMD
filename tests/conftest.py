import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onpolicy.config import get_config  # noqa: E402
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # noqa: E402

# Ground truth captured from live StarCraft II environments on branch 5-28.
# Layout: [total, [n_allies, ally_dim], [n_enemies, enemy_dim],
#          [1, move_dim], [1, own+id+timestep]]
SMAC_STRUCTURES = {
    '3m': [64, [2, 14], [3, 5], [1, 4], [1, 17]],
    '1c3s5z': [310, [8, 24], [9, 9], [1, 4], [1, 33]],
    'MMM2': [370, [9, 26], [12, 8], [1, 4], [1, 36]],
}

# n_agents / n_enemies as declared by the map itself.
SMAC_MAP_COUNTS = {
    '3m': {'n_agents': 3, 'n_enemies': 3},
    '1c3s5z': {'n_agents': 9, 'n_enemies': 9},
    'MMM2': {'n_agents': 10, 'n_enemies': 12},
}


class DiscreteAction(object):
    """Minimal stand-in for gym.spaces.Discrete accepted by ACTLayer."""

    def __init__(self, n):
        self.n = n
        self.__class__.__name__ = 'Discrete'


def make_smac_args(extra=None, **overrides):
    argv = ['--use_hetero_graph', '--hidden_size', '64']
    argv += list(extra or [])
    args = get_config().parse_known_args(argv)[0]
    args.env_name = 'StarCraft2'
    args.scenario_name = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_smac_actor(map_name, extra=None, n_actions=None, **overrides):
    structure = SMAC_STRUCTURES[map_name]
    args = make_smac_args(extra=extra, **overrides)
    if n_actions is None:
        n_actions = 6 + structure[2][0]
    return R_Actor(args, structure, DiscreteAction(n_actions))


def make_mpe_actor(scenario_name, obs_dim, extra):
    argv = ['--use_hetero_graph', '--hidden_size', '64'] + list(extra)
    args = get_config().parse_known_args(argv)[0]
    args.env_name = 'MPE'
    args.scenario_name = scenario_name
    from gym import spaces
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,),
                           dtype=np.float32)
    return R_Actor(args, obs_space, spaces.Discrete(5))


def build_smac_obs(spec, batch, seed=0, ally_visible=None, enemy_visible=None):
    """Compose a syntheticly-structured SMAC observation.

    ``ally_visible`` / ``enemy_visible`` are [batch, n] bool arrays. Invisible
    units are written as all-zero blocks, matching the real environment.
    """
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen) * 0.8 + 0.1

    ally = rand(batch, spec.n_allies, spec.ally_feat_dim)
    enemy = rand(batch, spec.n_enemies, spec.enemy_feat_dim)

    if spec.n_allies:
        ally[..., 0] = 1.0
        if ally_visible is not None:
            hidden = ~torch.as_tensor(ally_visible, dtype=torch.bool)
            ally[hidden] = 0.0
    if spec.n_enemies:
        if enemy_visible is not None:
            hidden = ~torch.as_tensor(enemy_visible, dtype=torch.bool)
            enemy[hidden] = 0.0

    move = rand(batch, spec.move_feat_dim)
    own = rand(batch, spec.own_feat_dim)
    obs = torch.cat([ally.reshape(batch, -1), enemy.reshape(batch, -1),
                     move, own], dim=-1)
    assert obs.shape[-1] == spec.total_dim
    return obs, ally, enemy, move, own


@pytest.fixture(scope='session')
def smac_structures():
    return SMAC_STRUCTURES
