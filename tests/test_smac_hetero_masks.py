"""Visibility masking for SMAC allies and enemies.

The parser returns ``*_mask`` with True = invalid, which the graph inverts
once via ``_mask_to_valid``.
"""

import pytest
import torch

from conftest import build_smac_obs, make_smac_actor


def test_invisible_ally_is_masked():
    actor = make_smac_actor('1c3s5z')
    spec = actor.smac_obs_spec
    visible = torch.ones(4, spec.n_allies, dtype=torch.bool)
    visible[0, 0] = False
    visible[1, 2] = False
    visible[3, :] = False

    obs, _, _, _, _ = build_smac_obs(spec, batch=4, ally_visible=visible)
    ally_mask = actor._split_smac_hetero_obs(obs)[4]

    assert ally_mask.dtype == torch.bool
    torch.testing.assert_close(~ally_mask, visible)


def test_invisible_enemy_is_masked():
    actor = make_smac_actor('MMM2')
    spec = actor.smac_obs_spec
    visible = torch.ones(3, spec.n_enemies, dtype=torch.bool)
    visible[0, 1] = False
    visible[1, :] = False

    obs, _, _, _, _ = build_smac_obs(spec, batch=3, enemy_visible=visible)
    enemy_mask = actor._split_smac_hetero_obs(obs)[5]

    torch.testing.assert_close(~enemy_mask, visible)


def test_visible_but_unattackable_enemy_stays_valid():
    """enemy_feats[0] is attack availability, not visibility.

    An enemy in sight but out of shoot range has feature 0 == 0 while the rest
    of its block is populated. It must remain a valid graph node.
    """
    actor = make_smac_actor('3m')
    spec = actor.smac_obs_spec
    obs, ally, enemy, move, own = build_smac_obs(spec, batch=2, seed=11)

    # Enemy 0: visible, cannot be attacked right now.
    enemy[:, 0, 0] = 0.0
    enemy[:, 0, 1] = 0.7   # distance
    enemy[:, 0, 2] = 0.3   # relative x
    enemy[:, 0, 3] = -0.2  # relative y
    # Enemy 1: genuinely invisible -> all zero.
    enemy[:, 1, :] = 0.0

    rebuilt = torch.cat([ally.reshape(2, -1), enemy.reshape(2, -1), move, own],
                        dim=-1)
    enemy_mask = actor._split_smac_hetero_obs(rebuilt)[5]

    assert not enemy_mask[:, 0].any(), 'unattackable enemy was wrongly masked'
    assert enemy_mask[:, 1].all(), 'invisible enemy should be masked'


def test_dead_ally_all_zero_block_is_masked():
    actor = make_smac_actor('3m')
    spec = actor.smac_obs_spec
    obs, ally, enemy, move, own = build_smac_obs(spec, batch=2, seed=13)
    ally[:, 0, :] = 0.0

    rebuilt = torch.cat([ally.reshape(2, -1), enemy.reshape(2, -1), move, own],
                        dim=-1)
    ally_mask = actor._split_smac_hetero_obs(rebuilt)[4]
    assert ally_mask[:, 0].all()


@pytest.mark.parametrize('map_name', ['3m', '1c3s5z', 'MMM2'])
def test_mask_convention_is_true_means_invalid(map_name):
    """Graph-side inversion must recover the intended validity exactly once."""
    from onpolicy.algorithms.utils.hetero_graph import HeteroGraphActorBase

    actor = make_smac_actor(map_name)
    spec = actor.smac_obs_spec
    visible = torch.ones(2, spec.n_allies, dtype=torch.bool)
    visible[0, 0] = False

    obs, _, _, _, _ = build_smac_obs(spec, batch=2, ally_visible=visible)
    ally_mask = actor._split_smac_hetero_obs(obs)[4]

    recovered = HeteroGraphActorBase._mask_to_valid(ally_mask)
    torch.testing.assert_close(recovered, visible)
