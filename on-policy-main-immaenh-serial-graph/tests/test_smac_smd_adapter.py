import os
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from gym.spaces import Box, Discrete

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, GraphRawObservationFusion
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.smac_obs_adapter import SMACHeteroObservationAdapter
from onpolicy.config import get_config


def make_metadata(
    n_allies=2,
    ally_dim=6,
    n_enemies=3,
    enemy_dim=7,
    move_dim=4,
    own_extra_dim=5,
):
    obs_dim = (
        n_allies * ally_dim
        + n_enemies * enemy_dim
        + move_dim
        + own_extra_dim
    )
    return [
        obs_dim,
        [n_allies, ally_dim],
        [n_enemies, enemy_dim],
        [1, move_dim],
        [1, own_extra_dim],
    ]


def make_flat_obs(metadata, batch_size=4):
    return torch.zeros(batch_size, metadata[0], dtype=torch.float32)


def entity_views(obs, metadata):
    n_allies, ally_dim = metadata[1]
    n_enemies, enemy_dim = metadata[2]
    move_dim = metadata[3][1]
    own_extra_dim = metadata[4][1]
    idx = 0
    ally = obs[:, idx:idx + n_allies * ally_dim].view(-1, n_allies, ally_dim)
    idx += n_allies * ally_dim
    enemy = obs[:, idx:idx + n_enemies * enemy_dim].view(-1, n_enemies, enemy_dim)
    idx += n_enemies * enemy_dim
    move = obs[:, idx:idx + move_dim]
    idx += move_dim
    own = obs[:, idx:idx + own_extra_dim]
    return ally, enemy, move, own


def make_args(**overrides):
    args = get_config().parse_args([])
    args.algorithm_name = "rmappo"
    args.env_name = "StarCraft2"
    args.map_name = "3m"
    args.hidden_size = 64
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.use_hetero_graph = True
    args.use_gated_fusion = True
    args.use_smd = True
    args.use_gsd_bsd = False
    args.use_knn = True
    args.use_am_filter = True
    args.smd_candidate_top_k = 2
    args.smd_edge_top_m = 1
    args.smd_pseudo_top_m = 1
    args.use_stacked_frames = False
    args.num_agents = 3
    args.num_enemies = 3
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_actor(metadata, **arg_overrides):
    return R_Actor(
        make_args(**arg_overrides),
        metadata,
        Discrete(9),
        torch.device("cpu"),
    )


def test_adapter_splits_expected_shapes():
    metadata = make_metadata()
    adapter = SMACHeteroObservationAdapter(metadata)

    outputs = adapter(make_flat_obs(metadata))
    agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask = outputs

    assert agent_state.shape == (4, 9)
    assert ally_obs.shape == (4, 2, 6)
    assert enemy_obs.shape == (4, 3, 7)
    assert ctx_obs.shape == (4, 0, 1)
    assert ally_mask.shape == (4, 2)
    assert enemy_mask.shape == (4, 3)
    assert ctx_mask.shape == (4, 0)
    assert ally_mask.dtype == torch.bool
    assert enemy_mask.dtype == torch.bool
    assert ctx_mask.dtype == torch.bool


def test_adapter_canonicalizes_ally_spatial_features():
    metadata = make_metadata()
    obs = make_flat_obs(metadata, batch_size=1)
    ally, _, _, _ = entity_views(obs, metadata)
    ally[0, 0, :4] = torch.tensor([1.0, 0.5, 0.3, -0.4])

    _, ally_obs, _, _, ally_mask, _, _ = SMACHeteroObservationAdapter(metadata)(obs)

    assert torch.allclose(ally_obs[0, 0, :4], torch.tensor([0.3, -0.4, 0.5, 1.0]))
    assert not ally_mask[0, 0]


def test_adapter_masks_invisible_ally():
    metadata = make_metadata()
    obs = make_flat_obs(metadata, batch_size=1)
    ally, _, _, _ = entity_views(obs, metadata)
    ally[0, 0, :4] = torch.tensor([0.0, 0.5, 0.3, -0.4])

    _, _, _, _, ally_mask, _, _ = SMACHeteroObservationAdapter(metadata)(obs)

    assert ally_mask[0, 0]


def test_adapter_keeps_visible_non_attackable_enemy():
    metadata = make_metadata()
    obs = make_flat_obs(metadata, batch_size=1)
    _, enemy, _, _ = entity_views(obs, metadata)
    enemy[0, 0, :5] = torch.tensor([0.0, 0.6, 0.1, -0.2, 0.8])

    _, _, enemy_obs, _, _, enemy_mask, _ = SMACHeteroObservationAdapter(metadata)(obs)

    assert torch.allclose(enemy_obs[0, 0, :4], torch.tensor([0.1, -0.2, 0.6, 0.0]))
    assert not enemy_mask[0, 0]


def test_adapter_masks_zero_enemy():
    metadata = make_metadata()
    obs = make_flat_obs(metadata, batch_size=1)

    _, _, _, _, _, enemy_mask, _ = SMACHeteroObservationAdapter(metadata)(obs)

    assert enemy_mask.all()


def test_adapter_fails_fast_on_observation_size_mismatch():
    metadata = make_metadata()
    adapter = SMACHeteroObservationAdapter(metadata)

    with pytest.raises(ValueError, match="observation size"):
        adapter(torch.zeros(2, metadata[0] + 1))


def test_all_invalid_entities_have_finite_graph_outputs():
    metadata = make_metadata()
    actor = make_actor(metadata)

    features, latent = actor._extract_features(make_flat_obs(metadata))

    assert features.shape == (4, 64)
    assert torch.isfinite(features).all()
    assert torch.isfinite(latent).all()


def test_smd_aux_and_backward_reach_student_and_teacher():
    metadata = make_metadata()
    args = make_args()
    policy = R_MAPPOPolicy(
        args,
        metadata,
        Box(low=-np.inf, high=np.inf, shape=(81,), dtype=np.float32),
        Discrete(9),
        torch.device("cpu"),
    )
    trainer = R_MAPPO(args, policy, torch.device("cpu"))
    obs = make_flat_obs(metadata, batch_size=6)
    ally, enemy, move, own = entity_views(obs, metadata)
    ally[:, 0, :5] = torch.tensor([1.0, 0.4, 0.2, -0.1, 0.9])
    ally[:, 1, :5] = torch.tensor([1.0, 0.8, -0.3, 0.2, 0.7])
    enemy[:, 0, :5] = torch.tensor([1.0, 0.3, 0.1, 0.2, 1.0])
    move[:] = 1.0
    own[:] = 0.5

    policy.actor._extract_features(obs)
    aux = policy.actor.base.last_smd_aux
    loss, info = trainer.compute_smd_aux_loss(aux, torch.ones(6, 1))
    policy.actor.zero_grad()
    loss.backward()

    expected_keys = {
        "edge_logits",
        "edge_probs",
        "hard_mask",
        "candidate_mask",
        "edge_context",
    }
    assert expected_keys <= aux.keys()
    assert aux["edge_logits"].shape == (6, 2)
    assert aux["edge_probs"].shape == (6, 2)
    assert aux["hard_mask"].shape == (6, 2)
    assert aux["candidate_mask"].shape == (6, 2)
    assert aux["edge_context"].shape == (6, 2, 64 * 4 + 3)
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in info.values())

    student_grad = sum(
        float(param.grad.abs().sum())
        for param in policy.actor.base.student_mask_head.parameters()
        if param.grad is not None
    )
    teacher_grad = sum(
        float(param.grad.abs().sum())
        for param in policy.actor.base.smd_teacher.parameters()
        if param.grad is not None
    )
    optimizer_param_ids = {
        id(param)
        for group in policy.actor_optimizer.param_groups
        for param in group["params"]
    }

    assert student_grad > 0.0
    assert teacher_grad > 0.0
    assert all(
        id(param) in optimizer_param_ids
        for param in policy.actor.base.student_mask_head.parameters()
    )
    assert all(
        id(param) in optimizer_param_ids
        for param in policy.actor.base.smd_teacher.parameters()
    )


def test_actor_uses_metadata_dimensions_and_preserves_available_actions():
    metadata = [64, [2, 14], [3, 5], [1, 4], [1, 17]]
    actor = make_actor(metadata)
    obs = np.zeros((3, metadata[0]), dtype=np.float32)
    available_actions = np.zeros((3, 9), dtype=np.float32)
    available_actions[:, 7] = 1.0
    rnn_states = np.zeros((3, 1, 64), dtype=np.float32)
    masks = np.ones((3, 1), dtype=np.float32)

    actions, log_probs, _, _ = actor(
        obs,
        rnn_states,
        masks,
        available_actions=available_actions,
        deterministic=True,
    )

    assert actor.smac_obs_dim == 64
    assert actor.smac_n_allies == 2
    assert actor.smac_ally_raw_dim == 14
    assert actor.smac_n_enemies == 3
    assert actor.smac_enemy_raw_dim == 5
    assert actor.smac_move_dim == 4
    assert actor.smac_own_extra_dim == 17
    assert actor.agent_state_dim == 17
    assert actor.base.ally_dim == 14
    assert actor.base.enemy_dim == 5
    assert torch.equal(actions, torch.full_like(actions, 7))
    assert torch.isfinite(log_probs).all()


def test_stacked_frames_are_rejected_for_smac_heterograph():
    metadata = make_metadata()

    with pytest.raises(NotImplementedError, match="stacked_frames"):
        make_actor(metadata, use_stacked_frames=True, stacked_frames=2)


@pytest.mark.parametrize("map_name", ["3m", "1c3s5z", "MMM2"])
def test_real_smac_map_metadata_runs_through_smd_actor(map_name):
    from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
    from onpolicy.scripts.train.train_smac import parse_args

    env_args = parse_args(["--map_name", map_name], get_config())
    env = StarCraft2Env(env_args)
    metadata = env.get_obs_size()
    actor = make_actor(metadata, map_name=map_name,
                       num_agents=env.n_agents, num_enemies=env.n_enemies,
                       use_graph_raw_obs_fusion=True)
    obs = make_flat_obs(metadata, batch_size=2)
    features, _ = actor._extract_features(obs)
    assert features.shape == (2, actor.hidden_size)
    assert torch.isfinite(features).all()
    assert actor.smac_ally_dim == metadata[1][1]
    assert actor.smac_enemy_dim == metadata[2][1]
    assert actor.raw_obs_fusion is not None
    assert 0.0 < float(actor.last_raw_obs_gate_mean) < 1.0


def test_raw_observation_fusion_changes_actor_feature_without_changing_shape():
    torch.manual_seed(3)
    fusion = GraphRawObservationFusion(obs_dim=11, hidden_size=16)
    graph = torch.randn(4, 16)
    raw_a = torch.zeros(4, 11)
    raw_b = torch.ones(4, 11)
    out_a = fusion(graph, raw_a)
    out_b = fusion(graph, raw_b)
    assert out_a.shape == graph.shape
    assert not torch.allclose(out_a, out_b)


def test_ppo_only_gradient_reaches_student_and_policy_path():
    torch.manual_seed(7)
    metadata = make_metadata()
    actor = make_actor(metadata, use_recurrent_policy=True,
                       use_smd_diffusion_teacher=False,
                       use_graph_raw_obs_fusion=True)
    obs = make_flat_obs(metadata, batch_size=6)
    ally, enemy, move, own = entity_views(obs, metadata)
    ally[:, 0, :4] = torch.tensor([1.0, 0.4, 0.2, -0.1])
    ally[:, 1, :4] = torch.tensor([1.0, 0.8, -0.3, 0.2])
    enemy[:, 0, :4] = torch.tensor([1.0, 0.3, 0.1, 0.2])
    move[:] = torch.randn_like(move)
    own[:] = torch.randn_like(own)
    actions = torch.zeros(6, 1)
    states = torch.zeros(6, 1, actor.hidden_size)
    masks = torch.ones(6, 1)
    with torch.no_grad():
        actor(obs, states, masks, deterministic=True)
        rollout_aux = {key: actor.base.last_smd_aux[key].clone()
                       for key in ("candidate_indices", "candidate_mask", "hard_mask")}
    log_probs, _, _ = actor.evaluate_actions_with_smd(obs, states, actions, masks)
    for key, expected in rollout_aux.items():
        assert torch.equal(actor.base.last_smd_aux[key], expected), key
    ratio = torch.exp(log_probs - (log_probs.detach() + 0.1))
    advantage = torch.ones_like(ratio)
    loss = -torch.min(ratio * advantage,
                      ratio.clamp(0.8, 1.2) * advantage).mean()
    actor.zero_grad()
    loss.backward()

    for module in (actor.base.student_mask_head.edge_mlp,
                   actor.base.linear_q3, actor.base.linear_k3,
                   actor.base.linear_v3, actor.base.linear_q2,
                   actor.raw_obs_fusion, actor.rnn, actor.act):
        norm = sum(float(p.grad.detach().norm()) for p in module.parameters()
                   if p.grad is not None)
        assert norm > 0.0, type(module).__name__


def test_smd_full_ppo_update_is_finite():
    metadata = make_metadata()
    args = make_args(use_smd_diffusion_teacher=True,
                     use_graph_raw_obs_fusion=True)
    policy = R_MAPPOPolicy(
        args, metadata,
        Box(low=-np.inf, high=np.inf, shape=(81,), dtype=np.float32),
        Discrete(9), torch.device("cpu"),
    )
    trainer = R_MAPPO(args, policy, torch.device("cpu"))
    obs = make_flat_obs(metadata, batch_size=6)
    ally, enemy, move, own = entity_views(obs, metadata)
    ally[:, 0, :4] = torch.tensor([1.0, 0.4, 0.2, -0.1])
    ally[:, 1, :4] = torch.tensor([1.0, 0.8, -0.3, 0.2])
    enemy[:, 0, :4] = torch.tensor([1.0, 0.3, 0.1, 0.2])
    move[:] = 0.5
    own[:] = 0.5
    sample = (
        torch.zeros(6, 81), obs, torch.zeros(6, 1, 64),
        torch.zeros(6, 1, 64), torch.zeros(6, 1),
        torch.zeros(6, 1), torch.ones(6, 1), torch.ones(6, 1),
        torch.ones(6, 1), torch.zeros(6, 1), torch.ones(6, 1),
        torch.ones(6, 9),
    )
    result = trainer.ppo_update(sample)
    assert all(torch.isfinite(x).all() for x in result[:6] if torch.is_tensor(x))
    assert np.isfinite(result[-1]['student_grad_norm'])
    assert result[-1]['student_grad_norm'] > 0.0
    assert result[-1]['raw_obs_fusion_grad_norm'] > 0.0
    assert 0.0 < result[-1]['raw_obs_gate_mean'] < 1.0


def test_classic_smac_visibility_matrix_uses_supported_bool_dtype():
    from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env

    env = object.__new__(StarCraft2Env)
    ally = SimpleNamespace(
        health=1.0, pos=SimpleNamespace(x=0.0, y=0.0)
    )
    enemy = SimpleNamespace(
        health=1.0, pos=SimpleNamespace(x=1.0, y=0.0)
    )
    env.n_agents = 1
    env.n_enemies = 1
    env.agents = {0: ally}
    env.enemies = {0: enemy}
    env.get_unit_by_id = lambda _: ally
    env.unit_sight_range = lambda _: 10.0
    env.distance = lambda x1, y1, x2, y2: abs(x2 - x1) + abs(y2 - y1)

    visibility = env.get_visibility_matrix()

    assert visibility.dtype == np.bool_
    assert visibility.tolist() == [[False, True]]
