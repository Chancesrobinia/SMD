"""SMACObsSpec is derived from environment metadata, never from map names."""

import os

import pytest

from conftest import SMAC_MAP_COUNTS, SMAC_STRUCTURES
from onpolicy.algorithms.utils.smac_obs_spec import SMACObsSpec


@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_metadata_is_auto_derived(map_name):
    spec = SMACObsSpec.from_obs_space(SMAC_STRUCTURES[map_name])
    structure = SMAC_STRUCTURES[map_name]

    assert spec.total_dim == structure[0]
    assert (spec.n_allies, spec.ally_feat_dim) == tuple(structure[1])
    assert (spec.n_enemies, spec.enemy_feat_dim) == tuple(structure[2])
    assert spec.move_feat_dim == structure[3][1]
    assert spec.own_feat_dim == structure[4][1]


@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_ally_count_is_agents_minus_one(map_name):
    spec = SMACObsSpec.from_obs_space(SMAC_STRUCTURES[map_name])
    assert spec.n_allies == SMAC_MAP_COUNTS[map_name]['n_agents'] - 1


@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_enemy_count_matches_map(map_name):
    spec = SMACObsSpec.from_obs_space(SMAC_STRUCTURES[map_name])
    assert spec.n_enemies == SMAC_MAP_COUNTS[map_name]['n_enemies']


@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_fields_sum_to_total(map_name):
    spec = SMACObsSpec.from_obs_space(SMAC_STRUCTURES[map_name])
    expected = (spec.n_allies * spec.ally_feat_dim
                + spec.n_enemies * spec.enemy_feat_dim
                + spec.move_feat_dim + spec.own_feat_dim)
    assert expected == spec.total_dim
    assert spec.agent_state_dim == spec.move_feat_dim + spec.own_feat_dim
    assert spec.ctx_dim == spec.move_feat_dim


@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_ally_and_enemy_dims_may_differ(map_name):
    """Heterogeneous widths must be representable, not collapsed."""
    spec = SMACObsSpec.from_obs_space(SMAC_STRUCTURES[map_name])
    assert spec.ally_feat_dim != spec.enemy_feat_dim


def test_mmm2_has_more_enemies_than_allies():
    spec = SMACObsSpec.from_obs_space(SMAC_STRUCTURES['MMM2'])
    assert spec.n_enemies > spec.n_allies


def test_inconsistent_metadata_is_rejected():
    with pytest.raises(ValueError):
        SMACObsSpec(total_dim=999, n_allies=2, ally_feat_dim=14,
                    n_enemies=3, enemy_feat_dim=5, move_feat_dim=4,
                    own_feat_dim=17)


def test_unstructured_obs_space_is_rejected():
    """A plain Box carries no field layout, so guessing is refused."""
    from gym import spaces
    import numpy as np
    box = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        SMACObsSpec.from_obs_space(box)


@pytest.mark.parametrize('env_name', ['SMAC', 'SMACv2', 'StarCraft2v2'])
def test_non_classic_envs_are_refused(env_name):
    """SMACv2 et al. must not silently reuse the classic parser."""
    with pytest.raises(NotImplementedError):
        SMACObsSpec.from_obs_space(SMAC_STRUCTURES['3m'], env_name=env_name)


@pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser('~/StarCraftII')),
    reason='StarCraft II installation not available')
@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_matches_live_environment(map_name):
    """The recorded structures must still match a real environment."""
    from onpolicy.config import get_config
    from onpolicy.scripts.train.train_smac import parse_args
    from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env

    os.environ.setdefault('SC2PATH', os.path.expanduser('~/StarCraftII'))
    args = parse_args(['--map_name', map_name], get_config())
    env = StarCraft2Env(args)
    live = env.get_obs_size()

    spec = SMACObsSpec.from_obs_space(live)
    assert spec.total_dim == live[0]
    assert (spec.n_allies, spec.ally_feat_dim) == tuple(live[1])
    assert (spec.n_enemies, spec.enemy_feat_dim) == tuple(live[2])
    assert spec.n_allies == env.n_agents - 1
    assert spec.n_enemies == env.n_enemies
    assert list(SMAC_STRUCTURES[map_name][1]) == list(live[1])
    assert list(SMAC_STRUCTURES[map_name][2]) == list(live[2])
