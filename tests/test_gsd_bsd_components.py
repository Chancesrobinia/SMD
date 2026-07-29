import os
import sys

import torch


PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "on-policy-main-immaenh-serial-graph",
)
sys.path.insert(0, PROJECT_ROOT)

from onpolicy.algorithms.utils.gsd_bsd import (  # noqa: E402
    apply_budget_operation,
    budget_topm_mask,
    build_budget_operation_distribution,
    corrupt_target_mask,
    operation_target_index,
    ppo_surrogate,
    select_best_target_mask,
    select_relation_candidates,
    straight_through_budget_mask,
)


def test_budget_topm_mask_preserves_valid_budget_and_padding():
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8], [0.7, 0.6, 0.5, 0.4]])
    valid = torch.tensor([[True, True, False, True], [False, True, True, False]])

    mask = budget_topm_mask(scores, valid, edge_m=2)

    assert torch.equal(mask, torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]))
    assert torch.equal(mask.bool() & ~valid, torch.zeros_like(valid))
    assert torch.equal(mask.sum(-1), torch.tensor([2.0, 2.0]))


def test_budget_topm_mask_handles_zero_and_full_budget():
    scores = torch.randn(2, 3)
    valid = torch.tensor([[True, False, True], [True, True, True]])

    zero = budget_topm_mask(scores, valid, edge_m=0)
    full = budget_topm_mask(scores, valid, edge_m=9)

    assert zero.shape == scores.shape
    assert zero.sum().item() == 0.0
    assert torch.equal(full, valid.float())


def test_budget_topm_mask_handles_empty_candidates_without_topk():
    scores = torch.zeros(3, 0)
    valid = torch.zeros(3, 0, dtype=torch.bool)

    mask = budget_topm_mask(scores, valid, edge_m=2)

    assert mask.shape == (3, 0)
    assert mask.sum().item() == 0.0


def test_operation_distribution_enumerates_only_legal_pairs_and_noop():
    current = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, True, False]])
    drop_logits = torch.tensor([[1.0, 10.0, 2.0, 10.0]])
    add_logits = torch.tensor([[10.0, 3.0, 10.0, 5.0]])
    noop_logits = torch.tensor([0.5])

    ops = build_budget_operation_distribution(drop_logits, add_logits, noop_logits, current, valid)

    assert ops["operation_logits"].shape == (1, 3)
    assert torch.equal(ops["drop_indices"][0], torch.tensor([0, 2, -1]))
    assert torch.equal(ops["add_indices"][0], torch.tensor([1, 1, -1]))
    assert torch.equal(ops["valid_operations"][0], torch.tensor([True, True, True]))
    assert torch.allclose(ops["operation_logits"][0], torch.tensor([4.0, 5.0, 0.5]))


def test_noop_when_no_legal_swap_for_zero_or_full_budget():
    valid = torch.tensor([[True, True, False]])
    for current in (torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]])):
        ops = build_budget_operation_distribution(
            torch.zeros_like(current),
            torch.zeros_like(current),
            torch.tensor([1.0]),
            current,
            valid,
        )
        assert ops["operation_logits"].shape == (1, 1)
        hard, soft = apply_budget_operation(current, valid, ops, torch.tensor([0]))
        assert torch.equal(hard, current)
        assert torch.equal(soft, current)


def test_hard_and_soft_swap_preserve_budget_and_padding():
    current = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, True, False]])
    ops = {
        "operation_logits": torch.tensor([[2.0, 0.0, -1.0]]),
        "drop_indices": torch.tensor([[0, 2, -1]]),
        "add_indices": torch.tensor([[1, 1, -1]]),
        "valid_operations": torch.tensor([[True, True, True]]),
    }

    hard, soft = apply_budget_operation(current, valid, ops, torch.tensor([0]))

    assert torch.equal(hard, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
    assert torch.allclose(soft.sum(-1), current.sum(-1), atol=1e-6)
    assert soft[0, 3].item() == 0.0


def test_st_forward_equals_hard_and_backward_reaches_operation_logits():
    current = torch.tensor([[1.0, 0.0, 1.0]])
    valid = torch.tensor([[True, True, True]])
    ops = {
        "operation_logits": torch.tensor([[0.1, 2.0, 0.0]], requires_grad=True),
        "drop_indices": torch.tensor([[0, 2, -1]]),
        "add_indices": torch.tensor([[1, 1, -1]]),
        "valid_operations": torch.tensor([[True, True, True]]),
    }

    hard, soft = apply_budget_operation(current, valid, ops, torch.tensor([1]))
    st = straight_through_budget_mask(hard, soft)
    loss = (st * torch.tensor([[0.3, -0.2, 0.7]])).sum()
    loss.backward()

    assert torch.equal(st, hard)
    assert ops["operation_logits"].grad is not None
    assert ops["operation_logits"].grad.abs().sum().item() > 0.0


def test_operation_target_index_and_corruption_reverse_operation():
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    corrupted = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    valid = torch.tensor([[True, True, True, True]])
    ops = build_budget_operation_distribution(
        torch.zeros_like(target),
        torch.zeros_like(target),
        torch.zeros(1),
        corrupted,
        valid,
    )

    idx = operation_target_index(ops, drop_index=torch.tensor([2]), add_index=torch.tensor([1]), noop=False)
    restored, _ = apply_budget_operation(corrupted, valid, ops, idx)

    assert idx.item() >= 0
    assert torch.equal(restored, target)

    gen = torch.Generator().manual_seed(4)
    out = corrupt_target_mask(target, valid, corruption_prob=1.0, generator=gen)
    reverse = operation_target_index(
        build_budget_operation_distribution(torch.zeros_like(target), torch.zeros_like(target), torch.zeros(1), out["corrupted_mask"], valid),
        out["reverse_drop_index"],
        out["reverse_add_index"],
        noop=False,
    )
    recovered, _ = apply_budget_operation(
        out["corrupted_mask"],
        valid,
        build_budget_operation_distribution(torch.zeros_like(target), torch.zeros_like(target), torch.zeros(1), out["corrupted_mask"], valid),
        reverse,
    )
    assert torch.equal(recovered, target)


def test_select_relation_candidates_is_vectorized_and_excludes_padding():
    features = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    rel = torch.tensor(
        [
            [[0.0, 0.0], [3.0, 0.0], [1.0, 0.0], [0.5, 0.0], [2.0, 0.0]],
            [[4.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.1, 0.0]],
        ]
    )
    valid = torch.tensor([[False, True, True, False, True], [True, True, False, True, False]])

    out = select_relation_candidates(features, rel, valid, top_k=2)

    assert out["candidate_features"].shape == (2, 2, 3)
    assert torch.equal(out["candidate_original_indices"], torch.tensor([[2, 4], [1, 3]]))
    assert torch.equal(out["candidate_valid_mask"], torch.tensor([[True, True], [True, True]]))


def test_ppo_surrogate_target_selection_handles_positive_and_negative_advantages():
    old = torch.zeros(2, 1)
    adv = torch.tensor([[1.0], [-1.0]])
    base_logp = torch.tensor([[0.0], [0.0]])
    candidate_logp = torch.tensor(
        [
            [[0.0], [0.1], [-0.1]],
            [[0.0], [0.1], [-0.1]],
        ]
    )
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )

    surrogate = ppo_surrogate(
        candidate_logp,
        old.unsqueeze(1).expand_as(candidate_logp),
        adv.unsqueeze(1).expand_as(candidate_logp),
        0.2,
    )
    chosen, info = select_best_target_mask(candidates, surrogate, base_index=0, margin=0.0)

    assert torch.equal(chosen[0], candidates[0, 1])
    assert torch.equal(chosen[1], candidates[1, 2])
    assert info["target_changed"].tolist() == [True, True]
