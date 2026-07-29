import os
import sys

import numpy as np
import torch
from gym.spaces import Box, Discrete


PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "on-policy-main-immaenh-serial-graph",
)
sys.path.insert(0, PROJECT_ROOT)

from onpolicy.config import get_config  # noqa: E402
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # noqa: E402


def make_args(**overrides):
    parser = get_config()
    args = parser.parse_args([])
    args.algorithm_name = "rmappo"
    args.hidden_size = 64
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.use_hetero_graph = True
    args.use_gated_fusion = True
    args.use_smd = False
    args.use_knn = True
    args.use_am_filter = True
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_actor(args, obs_dim, action_n=5):
    return R_Actor(
        args,
        Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        Discrete(action_n),
        torch.device("cpu"),
    )


def spread_args(**overrides):
    args = make_args(
        scenario_name="simple_spread",
        num_agents=10,
        num_landmarks=10,
        agent_state_dim=4,
        landmark_dim=2,
        neighbor_dim=4,
        k_max=8,
        top_k_filter=6,
        gsd_bsd_ally_candidate_k=4,
        gsd_bsd_enemy_candidate_k=0,
        gsd_bsd_ally_edge_m=2,
        gsd_bsd_enemy_edge_m=0,
        **overrides,
    )
    return args


def tag_args(**overrides):
    args = make_args(
        scenario_name="simple_tag",
        num_agents=9,
        num_good_agents=3,
        num_adversaries=6,
        num_landmarks=2,
        agent_state_dim=4,
        landmark_dim=2,
        neighbor_dim=5,
        k_max=5,
        top_k_filter=3,
        gsd_bsd_ally_candidate_k=4,
        gsd_bsd_enemy_candidate_k=3,
        gsd_bsd_ally_edge_m=2,
        gsd_bsd_enemy_edge_m=1,
        **overrides,
    )
    return args


def test_use_gsd_bsd_false_matches_baseline_outputs_with_same_weights():
    torch.manual_seed(1)
    base_args = spread_args(use_gsd_bsd=False)
    bsd_args = spread_args(use_gsd_bsd=False)
    actor_a = make_actor(base_args, obs_dim=60)
    actor_b = make_actor(bsd_args, obs_dim=60)
    actor_b.load_state_dict(actor_a.state_dict())
    obs = np.random.randn(4, 60).astype(np.float32)
    rnn = np.zeros((4, 1, 64), dtype=np.float32)
    masks = np.ones((4, 1), dtype=np.float32)

    with torch.no_grad():
        feat_a, latent_a = actor_a._extract_features(torch.from_numpy(obs))
        feat_b, latent_b = actor_b._extract_features(torch.from_numpy(obs))

    assert torch.allclose(feat_a, feat_b, atol=0.0, rtol=0.0)
    assert torch.allclose(latent_a, latent_b, atol=0.0, rtol=0.0)


def test_use_smd_and_gsd_bsd_are_mutually_exclusive():
    args = spread_args(use_smd=True, use_gsd_bsd=True)
    try:
        make_actor(args, obs_dim=60)
    except RuntimeError as exc:
        assert "use_smd" in str(exc) and "use_gsd_bsd" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for SMD/BSD conflict")


def test_spread_bsd_has_no_enemy_branch_and_preserves_ally_budget():
    args = spread_args(use_gsd_bsd=True)
    actor = make_actor(args, obs_dim=60)
    obs = torch.randn(5, 60)

    features, latent = actor._extract_features(obs)
    aux = actor.base.last_gsd_bsd_aux

    assert features.shape == (5, 64)
    assert latent.shape[0] == 5
    assert aux["enemy_final_mask"].shape == (5, 0)
    assert aux["enemy_diffusion_loss"].item() == 0.0
    assert torch.equal(aux["ally_final_mask"].sum(-1), torch.full((5,), 2.0))
    assert aux["ally_soft_budget_error"].max().item() < 1e-5


def test_tag_bsd_rollout_and_evaluate_feature_path_is_deterministic():
    args = tag_args(use_gsd_bsd=True)
    actor = make_actor(args, obs_dim=30)
    obs = torch.randn(4, 30)

    with torch.no_grad():
        feat_rollout, _ = actor._extract_features(obs)
        feat_eval, _ = actor._extract_features(obs)

    assert torch.allclose(feat_rollout, feat_eval, atol=0.0, rtol=0.0)


def test_ppo_loss_reaches_bsd_denoiser_through_st_mask():
    args = tag_args(use_gsd_bsd=True, gsd_bsd_use_st_mask=True)
    actor = make_actor(args, obs_dim=30)
    obs = torch.randn(6, 30)
    rnn = np.zeros((6, 1, 64), dtype=np.float32)
    masks = np.ones((6, 1), dtype=np.float32)

    action, _, _, _ = actor(obs.numpy(), rnn, masks, deterministic=True)
    logp, _ = actor.evaluate_actions(obs.numpy(), rnn, action.detach().numpy(), masks)
    loss = -logp.mean()
    actor.zero_grad()
    loss.backward()

    grad = 0.0
    for param in actor.base.gsd_bsd_denoiser.parameters():
        if param.grad is not None:
            grad += float(param.grad.abs().sum().detach())
    assert grad > 0.0
