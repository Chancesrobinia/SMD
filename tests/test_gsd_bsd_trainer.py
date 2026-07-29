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
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO  # noqa: E402


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
    args.use_gsd_bsd = True
    args.use_valuenorm = False
    args.use_popart = False
    args.num_agents = 9
    args.num_good_agents = 3
    args.num_adversaries = 6
    args.num_landmarks = 2
    args.scenario_name = "simple_tag"
    args.agent_state_dim = 4
    args.landmark_dim = 2
    args.neighbor_dim = 5
    args.k_max = 5
    args.top_k_filter = 3
    args.gsd_bsd_ally_candidate_k = 4
    args.gsd_bsd_enemy_candidate_k = 3
    args.gsd_bsd_ally_edge_m = 2
    args.gsd_bsd_enemy_edge_m = 1
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_compute_gsd_bsd_aux_loss_backpropagates_to_denoiser():
    args = make_args()
    policy = R_MAPPOPolicy(
        args,
        Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(30 * args.num_agents,), dtype=np.float32),
        Discrete(5),
        torch.device("cpu"),
    )
    trainer = R_MAPPO(args, policy, torch.device("cpu"))
    obs = torch.randn(6, 30)
    rnn = np.zeros((6, 1, 64), dtype=np.float32)
    masks = np.ones((6, 1), dtype=np.float32)

    policy.actor._extract_features(obs)
    aux = policy.actor.base.last_gsd_bsd_aux
    loss, info = trainer.compute_gsd_bsd_aux_loss(aux)
    policy.actor.zero_grad()
    loss.backward()

    grad = 0.0
    for param in policy.actor.base.gsd_bsd_denoiser.parameters():
        if param.grad is not None:
            grad += float(param.grad.abs().sum())

    assert loss.item() >= 0.0
    assert info["gsd_bsd/total_diffusion_loss"] >= 0.0
    assert grad > 0.0


def test_compute_gsd_bsd_aux_loss_skips_missing_enemy():
    args = make_args(
        scenario_name="simple_spread",
        num_agents=10,
        num_landmarks=10,
        agent_state_dim=4,
        landmark_dim=2,
        neighbor_dim=4,
        gsd_bsd_enemy_candidate_k=0,
        gsd_bsd_enemy_edge_m=0,
    )
    policy = R_MAPPOPolicy(
        args,
        Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
        Box(low=-np.inf, high=np.inf, shape=(60 * args.num_agents,), dtype=np.float32),
        Discrete(5),
        torch.device("cpu"),
    )
    trainer = R_MAPPO(args, policy, torch.device("cpu"))
    policy.actor._extract_features(torch.randn(4, 60))
    loss, info = trainer.compute_gsd_bsd_aux_loss(policy.actor.base.last_gsd_bsd_aux)

    assert info["gsd_bsd/enemy_diffusion_loss"] == 0.0
    assert torch.isfinite(loss)
