"""Graph layers must be built with true ally/enemy/context widths."""

import pytest
import torch

from conftest import SMAC_STRUCTURES, build_smac_obs, make_smac_actor


MAPS = ['3m', '1c3s5z', 'MMM2']


@pytest.mark.parametrize('map_name', MAPS)
def test_graph_uses_heterogeneous_widths(map_name):
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec

    assert actor.base.ally_dim == spec.ally_feat_dim
    assert actor.base.enemy_dim == spec.enemy_feat_dim
    assert actor.base.ctx_dim == spec.ctx_dim
    assert actor.base.agent_state_dim == spec.agent_state_dim


@pytest.mark.parametrize('map_name', MAPS)
def test_layer_shapes_match_spec(map_name):
    """The projections must accept the real feature widths."""
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec

    assert actor.base.linear_k3.in_features == spec.ally_feat_dim
    assert actor.base.linear_k2_enemy.in_features == spec.enemy_feat_dim
    assert actor.base.linear_k2_ctx.in_features == spec.ctx_dim
    assert actor.base.ally_perception.k_proj.in_features == spec.ally_feat_dim


@pytest.mark.parametrize('map_name', MAPS)
def test_neighbor_dim_does_not_drive_smac(map_name):
    """neighbor_dim stays an MPE/ESMG concept and must not size SMAC tensors."""
    actor = make_smac_actor(map_name, extra=['--neighbor_dim', '8'])
    spec = actor.smac_obs_spec
    assert actor.base.ally_dim == spec.ally_feat_dim
    assert actor.base.enemy_dim == spec.enemy_feat_dim
    assert actor.base.ally_dim != 8 or spec.ally_feat_dim == 8


@pytest.mark.parametrize('map_name', MAPS)
def test_forward_runs_without_reshape_error(map_name):
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    obs, _, _, _, _ = build_smac_obs(spec, batch=6)

    features, _ = actor._extract_features(obs)
    assert features.shape == (6, actor.hidden_size)
    assert torch.isfinite(features).all()


def test_stacked_frames_are_refused():
    with pytest.raises(RuntimeError, match='stacked'):
        make_smac_actor('3m', use_stacked_frames=True, stacked_frames=4)


@pytest.mark.parametrize('env_name', ['SMAC', 'SMACv2', 'StarCraft2v2'])
def test_unverified_smac_variants_are_refused(env_name):
    with pytest.raises(NotImplementedError):
        make_smac_actor('3m', env_name=env_name)
