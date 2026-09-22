"""PPO loss must reach synergy_mlp through the sparse Top-K selector.

Before the fix, synergy scores went through torch.topk -> integer indices ->
gather, so synergy_mlp.weight.grad was None for every configuration.
"""

import pytest
import torch

from conftest import build_smac_obs, make_smac_actor


def synergy_linears(actor):
    return [m for m in actor.base.synergy_mlp.modules()
            if isinstance(m, torch.nn.Linear)]


def ppo_like_backward(actor, obs, n_actions):
    """Minimal PPO surrogate through the real training entry point."""
    batch = obs.shape[0]
    rnn_states = torch.zeros(batch, actor._recurrent_N, actor.hidden_size)
    masks = torch.ones(batch, 1)

    with torch.no_grad():
        actions = actor(obs, rnn_states, masks)[0]

    action_log_probs, _ = actor.evaluate_actions(obs, rnn_states, actions, masks)
    old_log_probs = action_log_probs.detach() - 0.1
    advantages = torch.randn(batch, 1, generator=torch.Generator().manual_seed(0))

    ratio = torch.exp(action_log_probs - old_log_probs)
    surrogate = -torch.min(ratio * advantages,
                           torch.clamp(ratio, 0.9, 1.1) * advantages).mean()
    surrogate.backward()
    return surrogate


@pytest.mark.parametrize('top_k_filter', [1, 2])
def test_synergy_mlp_receives_nonzero_gradient(top_k_filter):
    """Top-1 is the degenerate case most likely to kill the gradient."""
    actor = make_smac_actor('1c3s5z',
                            extra=['--top_k_filter', str(top_k_filter),
                                   '--k_max', '4'])
    spec = actor.smac_obs_spec
    obs, _, _, _, _ = build_smac_obs(spec, batch=4, seed=21)

    loss = ppo_like_backward(actor, obs, 6 + spec.n_enemies)
    assert torch.isfinite(loss)

    layers = synergy_linears(actor)
    first, last = layers[0], layers[-1]
    for tag, layer in (('first', first), ('last', last)):
        assert layer.weight.grad is not None, \
            'synergy_mlp {} layer received no gradient'.format(tag)
        assert layer.weight.grad.abs().sum().item() > 0.0, \
            'synergy_mlp {} layer gradient is all zero'.format(tag)


@pytest.mark.parametrize('top_k_filter', [1, 2])
def test_gradient_flows_with_batch_and_allies_at_least_four(top_k_filter):
    actor = make_smac_actor('MMM2',
                            extra=['--top_k_filter', str(top_k_filter),
                                   '--k_max', '5'])
    spec = actor.smac_obs_spec
    assert spec.n_allies >= 4
    obs, _, _, _, _ = build_smac_obs(spec, batch=4, seed=22)

    ppo_like_backward(actor, obs, 6 + spec.n_enemies)
    for layer in synergy_linears(actor):
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum().item() > 0.0


def test_all_actor_gradients_are_finite():
    actor = make_smac_actor('3m', extra=['--top_k_filter', '2', '--k_max', '2'])
    spec = actor.smac_obs_spec
    obs, _, _, _, _ = build_smac_obs(spec, batch=4, seed=23)

    ppo_like_backward(actor, obs, 6 + spec.n_enemies)
    for name, param in actor.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                'non-finite gradient in {}'.format(name)


def test_no_grad_rollout_still_uses_hard_mask():
    """Inference path keeps exact hard Top-K semantics."""
    actor = make_smac_actor('1c3s5z', extra=['--top_k_filter', '2', '--k_max', '4'])
    spec = actor.smac_obs_spec
    obs, _, _, _, _ = build_smac_obs(spec, batch=4, seed=24)

    with torch.no_grad():
        actor._extract_features(obs)
    hard = actor.base.last_synergy_hard_mask
    assert set(hard.unique().tolist()) <= {0.0, 1.0}
