"""Straight-through Top-K must preserve hard forward semantics."""

import pytest
import torch

from onpolicy.algorithms.utils.hetero_graph import HeteroGraphActorBase


def make_graph(top_k_filter=2, k_max=4, ally_dim=6, **kwargs):
    torch.manual_seed(0)
    return HeteroGraphActorBase(
        agent_state_dim=12, landmark_dim=2, neighbor_dim=4,
        ally_dim=ally_dim, enemy_dim=5, ctx_dim=4, hidden_size=32,
        k_max=k_max, top_k_filter=top_k_filter,
        use_knn=True, use_am_filter=True, **kwargs)


def test_hard_mask_selects_exactly_top_k():
    graph = make_graph(top_k_filter=2)
    logits = torch.tensor([[5.0, 1.0, 4.0, 0.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)

    st_mask, hard = graph._straight_through_topk(logits, valid, 2)
    assert hard.sum().item() == 2
    # Highest two logits are indices 0 and 2.
    assert hard[0, 0] == 1 and hard[0, 2] == 1
    assert hard[0, 1] == 0 and hard[0, 3] == 0
    torch.testing.assert_close(st_mask, hard)


def test_forward_value_equals_hard_mask():
    """st_mask must be numerically identical to hard in the forward pass."""
    graph = make_graph(top_k_filter=2)
    logits = torch.randn(8, 4, requires_grad=True)
    valid = torch.ones(8, 4, dtype=torch.bool)

    st_mask, hard = graph._straight_through_topk(logits, valid, 2)
    torch.testing.assert_close(st_mask, hard)
    assert set(st_mask.detach().unique().tolist()) <= {0.0, 1.0}


def test_invalid_candidates_never_selected():
    graph = make_graph(top_k_filter=2)
    # Index 1 has the largest logit but is invalid.
    logits = torch.tensor([[1.0, 99.0, 2.0, 0.5]])
    valid = torch.tensor([[True, False, True, True]])

    _, hard = graph._straight_through_topk(logits, valid, 2)
    assert hard[0, 1] == 0
    assert hard.sum().item() == 2


def test_budget_shrinks_when_fewer_valid_than_k():
    graph = make_graph(top_k_filter=3)
    logits = torch.randn(4, 5)
    valid = torch.zeros(4, 5, dtype=torch.bool)
    valid[:, 0] = True  # only one valid candidate

    _, hard = graph._straight_through_topk(logits, valid, 3)
    assert hard.sum(dim=-1).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_no_valid_candidate_yields_empty_selection_without_nan():
    graph = make_graph(top_k_filter=2)
    logits = torch.randn(3, 4)
    valid = torch.zeros(3, 4, dtype=torch.bool)

    st_mask, hard = graph._straight_through_topk(logits, valid, 2)
    assert hard.sum().item() == 0
    assert torch.isfinite(st_mask).all()


def test_soft_mask_sums_to_budget():
    graph = make_graph(top_k_filter=2)
    logits = torch.randn(16, 6, generator=torch.Generator().manual_seed(4))
    valid = torch.ones(16, 6, dtype=torch.bool)
    budget = torch.full((16,), 2, dtype=torch.long)

    soft = graph._budgeted_soft_mask(logits, valid, budget, temperature=1.0)
    assert (soft >= 0).all() and (soft <= 1).all()
    torch.testing.assert_close(soft.sum(dim=-1),
                               torch.full((16,), 2.0), atol=1e-3, rtol=1e-3)


def test_soft_mask_zero_on_padding():
    graph = make_graph()
    logits = torch.randn(4, 5)
    valid = torch.tensor([[True, True, False, False, False]] * 4)
    budget = torch.full((4,), 2, dtype=torch.long)

    soft = graph._budgeted_soft_mask(logits, valid, budget)
    assert soft[:, 2:].abs().max().item() == 0.0


def test_no_valid_ally_gives_zero_coop_without_nan():
    """Requirement: m_coop = 0 and no NaN when nothing is selectable."""
    graph = make_graph(top_k_filter=2)
    batch = 4
    agent_state = torch.randn(batch, 12)
    ally = torch.zeros(batch, 5, 6)
    enemy = torch.randn(batch, 3, 5)
    ctx = torch.randn(batch, 1, 4)

    ally_mask = torch.ones(batch, 5, dtype=torch.bool)   # all invalid
    enemy_mask = torch.zeros(batch, 3, dtype=torch.bool)

    features, _ = graph(agent_state, ally, enemy, ctx,
                        ally_mask, enemy_mask, torch.ones(batch, 1),
                        ally_distance=ally[..., 1])
    assert torch.isfinite(features).all()
    assert graph.last_synergy_hard_mask.sum().item() == 0


@pytest.mark.parametrize('top_k_filter', [1, 2])
def test_selection_count_never_exceeds_budget(top_k_filter):
    graph = make_graph(top_k_filter=top_k_filter, k_max=4)
    logits = torch.randn(32, 4, generator=torch.Generator().manual_seed(9))
    valid = torch.rand(32, 4, generator=torch.Generator().manual_seed(10)) > 0.3

    _, hard = graph._straight_through_topk(logits, valid, top_k_filter)
    counts = hard.sum(dim=-1)
    valid_counts = valid.sum(dim=-1).to(counts.dtype)
    assert (counts <= top_k_filter).all()
    assert (counts <= valid_counts).all()
    assert (hard.bool() & ~valid).sum().item() == 0
