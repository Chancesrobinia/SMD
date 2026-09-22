"""The SMAC parser must consume the observation exactly, with no overlap."""

import pytest
import torch

from conftest import SMAC_STRUCTURES, build_smac_obs, make_smac_actor


MAPS = ['3m', '1c3s5z', 'MMM2']


@pytest.mark.parametrize('map_name', MAPS)
def test_split_shapes_follow_spec(map_name):
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    obs, _, _, _, _ = build_smac_obs(spec, batch=4)

    (agent_state, ally_obs, enemy_obs, ctx_obs,
     ally_mask, enemy_mask, ctx_mask, ally_distance) = actor._split_smac_hetero_obs(obs)

    assert ally_obs.shape == (4, spec.n_allies, spec.ally_feat_dim)
    assert enemy_obs.shape == (4, spec.n_enemies, spec.enemy_feat_dim)
    assert agent_state.shape == (4, spec.agent_state_dim)
    assert ctx_obs.shape == (4, 1, spec.ctx_dim)
    assert ally_mask.shape == (4, spec.n_allies)
    assert enemy_mask.shape == (4, spec.n_enemies)
    assert ctx_mask.shape == (4, 1)
    assert ally_distance.shape == (4, spec.n_allies)


@pytest.mark.parametrize('map_name', MAPS)
def test_parser_consumes_whole_observation(map_name):
    """Reassembling the split fields must reproduce the input exactly."""
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    obs, ally, enemy, move, own = build_smac_obs(spec, batch=4, seed=3)

    (agent_state, ally_obs, enemy_obs, ctx_obs,
     _, _, _, _) = actor._split_smac_hetero_obs(obs)

    torch.testing.assert_close(ally_obs, ally)
    torch.testing.assert_close(enemy_obs, enemy)
    torch.testing.assert_close(agent_state, torch.cat([move, own], dim=-1))
    torch.testing.assert_close(ctx_obs.squeeze(1), move)

    rebuilt = torch.cat([
        ally_obs.reshape(4, -1), enemy_obs.reshape(4, -1), agent_state,
    ], dim=-1)
    torch.testing.assert_close(rebuilt, obs)


@pytest.mark.parametrize('map_name', MAPS)
def test_fields_do_not_overlap(map_name):
    """Each observation slot must land in exactly one field."""
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    total = spec.total_dim
    obs = torch.arange(total, dtype=torch.float32).unsqueeze(0)

    (agent_state, ally_obs, enemy_obs, _,
     _, _, _, _) = actor._split_smac_hetero_obs(obs)

    seen = torch.cat([
        ally_obs.reshape(-1), enemy_obs.reshape(-1), agent_state.reshape(-1),
    ])
    assert seen.numel() == total
    assert sorted(seen.tolist()) == list(range(total))


@pytest.mark.parametrize('map_name', MAPS)
def test_context_is_move_features_not_a_proxy_slice(map_name):
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    obs, _, _, move, _ = build_smac_obs(spec, batch=2, seed=5)

    ctx_obs = actor._split_smac_hetero_obs(obs)[3]
    torch.testing.assert_close(ctx_obs.squeeze(1), move)
    assert ctx_obs.shape[-1] == spec.move_feat_dim


@pytest.mark.parametrize('map_name', MAPS)
def test_wrong_width_raises_instead_of_warning(map_name):
    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    bad = torch.zeros(2, spec.total_dim + 3)
    with pytest.raises(RuntimeError):
        actor._split_smac_hetero_obs(bad)


def test_ally_distance_uses_normalised_distance_channel():
    """KNN distance must be the env's distance feature, not feature 0/1 as dx,dy."""
    actor = make_smac_actor('3m')
    spec = actor.smac_obs_spec
    obs, ally, _, _, _ = build_smac_obs(spec, batch=4, seed=7)

    ally_distance = actor._split_smac_hetero_obs(obs)[7]
    torch.testing.assert_close(ally_distance, ally[..., 1])
