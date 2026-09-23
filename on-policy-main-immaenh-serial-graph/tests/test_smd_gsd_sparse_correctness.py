import torch

from onpolicy.algorithms.utils.sparse_mask_student import (
    straight_through_topm_mask,
)
from onpolicy.algorithms.utils.sparse_mask_diffusion import (
    SparseMaskDiffusionTeacher,
)
from onpolicy.algorithms.utils.smac_obs_adapter import SMACHeteroObservationAdapter


def test_straight_through_topm_has_budget_and_selector_gradients_only_for_partial_budget():
    logits = torch.tensor([[0.1, 0.8, 0.3]], requires_grad=True)
    valid = torch.tensor([[True, True, True]])
    mask, hard, soft = straight_through_topm_mask(logits, valid, edge_top_m=1)
    assert torch.equal(mask, hard)
    assert hard.sum().item() == 1.0
    (mask * torch.tensor([[1.0, 2.0, 4.0]])).sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0

    for budget in (0, 3):
        edge_logits = torch.randn(1, 3, requires_grad=True)
        mask, hard, soft = straight_through_topm_mask(
            edge_logits, valid, edge_top_m=budget
        )
        assert torch.equal(mask, hard)
        edge_logits.grad = None
        mask.sum().backward()
        assert edge_logits.grad is None or edge_logits.grad.abs().sum().item() == 0.0


def test_teacher_uses_one_timestep_per_sample_and_reconstructs_x0():
    teacher = SparseMaskDiffusionTeacher(
        edge_context_dim=11, hidden_dim=16, num_diffusion_steps=8
    )
    context = torch.randn(4, 3, 11)
    target = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float32)
    valid = torch.ones(4, 3, dtype=torch.bool)
    out = teacher(context, target, valid)
    assert out["timesteps"].shape == (4,)
    assert out["x0"].shape == target.shape
    assert out["x0_pred"].shape == target.shape
    assert torch.isfinite(out["x0_pred"]).all()


def test_smac_adapter_exposes_move_context_and_self_metadata():
    metadata = [26, [2, 4], [2, 5], [1, 3], [1, 5]]
    adapter = SMACHeteroObservationAdapter(metadata, include_move_context=True)
    obs = torch.zeros(2, 26)
    self_obs, allies, enemies, context, ally_mask, enemy_mask, context_mask = adapter(obs)
    assert adapter.n_allies == 2
    assert adapter.n_enemies == 2
    assert adapter.ally_dim == 4
    assert adapter.enemy_dim == 5
    assert adapter.move_dim == 3
    assert adapter.self_dim == 5
    assert self_obs.shape == (2, 5)
    assert context.shape == (2, 1, 3)
    assert context_mask.shape == (2, 1)
