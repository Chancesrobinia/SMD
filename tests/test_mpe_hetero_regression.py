"""MPE heterograph parsers must be untouched by the SMAC fixes.

The expected sums were captured by running the parsers on branch 5-28 at
7aff6d1 *before* any change, with torch.manual_seed(0) for construction and
torch.manual_seed(1) for the observation batch.
"""

import pytest
import torch

from conftest import make_mpe_actor


MPE_COMMON = ['--num_agents', '6', '--num_adversaries', '4',
              '--num_good_agents', '2', '--num_landmarks', '2',
              '--num_forests', '2']
TAG_ARGS = ['--agent_state_dim', '4', '--landmark_dim', '2',
            '--neighbor_dim', '5'] + MPE_COMMON
WORLD_ARGS = ['--agent_state_dim', '8', '--landmark_dim', '4',
              '--neighbor_dim', '5'] + MPE_COMMON
SPREAD_ARGS = ['--agent_state_dim', '4', '--landmark_dim', '2',
               '--neighbor_dim', '4', '--num_agents', '3']

FIELDS = ['agent_state', 'ally_obs', 'enemy_obs', 'ctx_obs',
          'ally_mask', 'enemy_mask', 'ctx_mask']

# scenario -> obs_dim -> {field: (shape, sum)}
GOLDEN = {
    ('simple_spread', 18): {
        'agent_state': ((4, 4), 2.434990),
        'ally_obs': ((4, 2, 4), 1.962532),
        'enemy_obs': ((4, 0, 4), 0.0),
        'ctx_obs': ((4, 3, 2), -5.472958),
        'ally_mask': ((4, 2), 0.0),
        'enemy_mask': ((4, 0), 0.0),
        'ctx_mask': ((4, 3), 12.0),
    },
    ('simple_tag', 22): {
        'agent_state': ((4, 4), -3.024298),
        'ally_obs': ((4, 3, 5), 17.209588),
        'enemy_obs': ((4, 2, 5), -13.470982),
        'ctx_obs': ((4, 2, 2), -8.595315),
        'ally_mask': ((4, 3), 0.0),
        'enemy_mask': ((4, 2), 0.0),
        'ctx_mask': ((4, 2), 8.0),
    },
    ('simple_tag', 20): {
        'agent_state': ((4, 4), -6.699085),
        'ally_obs': ((4, 1, 5), -6.399461),
        'enemy_obs': ((4, 4, 5), 18.809494),
        'ctx_obs': ((4, 2, 2), -6.450322),
        'ally_mask': ((4, 1), 0.0),
        'enemy_mask': ((4, 4), 0.0),
        'ctx_mask': ((4, 2), 8.0),
    },
    ('simple_world_comm', 32): {
        'agent_state': ((4, 8), -8.116969),
        'ally_obs': ((4, 3, 5), 10.817697),
        'enemy_obs': ((4, 2, 5), -2.574685),
        'ctx_obs': ((4, 5, 4), -8.587481),
        'ally_mask': ((4, 3), 0.0),
        'enemy_mask': ((4, 2), 0.0),
        'ctx_mask': ((4, 5), 20.0),
    },
    ('simple_world_comm', 28): {
        'agent_state': ((4, 8), -8.702678),
        'ally_obs': ((4, 1, 5), -8.113136),
        'enemy_obs': ((4, 4, 5), 19.087016),
        'ctx_obs': ((4, 5, 4), -10.195629),
        'ally_mask': ((4, 1), 0.0),
        'enemy_mask': ((4, 4), 0.0),
        'ctx_mask': ((4, 5), 20.0),
    },
}

ARGS_FOR = {
    'simple_spread': SPREAD_ARGS,
    'simple_tag': TAG_ARGS,
    'simple_world_comm': WORLD_ARGS,
}


def run_parser(scenario, obs_dim):
    torch.manual_seed(0)
    actor = make_mpe_actor(scenario, obs_dim, ARGS_FOR[scenario])
    torch.manual_seed(1)
    obs = torch.randn(4, obs_dim)
    if scenario == 'simple_world_comm':
        out = actor._split_world_comm_hetero_obs(obs)
    elif scenario == 'simple_tag':
        out = actor._split_tag_hetero_obs(obs)
    else:
        out = actor._split_hetero_obs(obs)
    return dict(zip(FIELDS, out))


@pytest.mark.parametrize('case', sorted(GOLDEN.keys()))
def test_mpe_parser_output_unchanged(case):
    scenario, obs_dim = case
    parsed = run_parser(scenario, obs_dim)
    for field, (shape, expected_sum) in GOLDEN[case].items():
        tensor = parsed[field]
        assert tuple(tensor.shape) == shape, \
            '{} {} shape drifted'.format(scenario, field)
        actual = float(tensor.to(torch.float64).sum())
        assert actual == pytest.approx(expected_sum, abs=1e-5), \
            '{} {} values drifted'.format(scenario, field)


@pytest.mark.parametrize('case', sorted(GOLDEN.keys()))
def test_mpe_parsers_return_seven_fields(case):
    """MPE must keep the legacy 7-tuple; only SMAC gained ally_distance."""
    scenario, obs_dim = case
    torch.manual_seed(0)
    actor = make_mpe_actor(scenario, obs_dim, ARGS_FOR[scenario])
    obs = torch.randn(4, obs_dim)
    if scenario == 'simple_world_comm':
        out = actor._split_world_comm_hetero_obs(obs)
    elif scenario == 'simple_tag':
        out = actor._split_tag_hetero_obs(obs)
    else:
        out = actor._split_hetero_obs(obs)
    assert len(out) == 7


@pytest.mark.parametrize('case', sorted(GOLDEN.keys()))
def test_mpe_forward_still_runs(case):
    scenario, obs_dim = case
    torch.manual_seed(0)
    actor = make_mpe_actor(scenario, obs_dim, ARGS_FOR[scenario])
    obs = torch.randn(4, obs_dim)
    features, _ = actor._extract_features(obs)
    assert features.shape == (4, actor.hidden_size)
    assert torch.isfinite(features).all()


def test_mpe_uses_neighbor_dim_for_ally_and_enemy():
    """neighbor_dim must still drive MPE ally/enemy widths."""
    torch.manual_seed(0)
    actor = make_mpe_actor('simple_tag', 22, TAG_ARGS)
    assert actor.base.ally_dim == 5
    assert actor.base.enemy_dim == 5
    assert actor.base.ctx_dim == 2
    assert not hasattr(actor, 'smac_obs_spec')
