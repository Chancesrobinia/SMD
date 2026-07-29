import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.esmg import EgoSparseMultiplexGraph
from onpolicy.algorithms.utils.hetero_graph import (
    EntityEnemyFirstGraphActorBase,
    HeteroGraphActorBase,
    ParallelAllyEntityGraphActorBase,
)
from onpolicy.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    Supports three base-network modes (mutually exclusive, checked in priority order):
      1. HeteroGraph  — if args.use_hetero_graph is True
      2. ESMG         — if args.self_dim and args.neighbor_dim are both set
      3. MLPBase/CNN  — fallback
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        # ---- Mode flags (priority: hetero > esmg > mlp) ----
        self._use_hetero_graph = getattr(args, 'use_hetero_graph', False)
        self._use_parallel_ally_graph = getattr(args, 'use_parallel_ally_graph', False)
        self._use_entity_enemy_first_graph = getattr(args, 'use_entity_enemy_first_graph', False)
        self.scenario_name = getattr(args, 'scenario_name', None)
        self._use_world_comm_hetero = (
            self._use_hetero_graph and self.scenario_name == 'simple_world_comm'
        )
        self._use_tag_hetero = (
            self._use_hetero_graph and self.scenario_name == 'simple_tag'
        )
        self._use_esmg = (
            not self._use_hetero_graph
            and getattr(args, 'self_dim', None) is not None
            and getattr(args, 'neighbor_dim', None) is not None
        )

        # Ablation switches
        self._use_knn = getattr(args, 'use_knn', True)
        self._use_am_filter = getattr(args, 'use_am_filter', True)
        self._use_smd = getattr(args, 'use_smd', False)
        self._use_gsd_bsd = getattr(args, 'use_gsd_bsd', False)
        if self._use_smd and self._use_gsd_bsd:
            raise RuntimeError("use_smd and use_gsd_bsd cannot be enabled together")
        self.last_graph_gates = None

        # ============================================================
        # Mode A: Heterogeneous Dual-Graph
        # ============================================================
        if self._use_hetero_graph:
            self.agent_state_dim = getattr(args, 'agent_state_dim', 4)
            self.landmark_dim    = getattr(args, 'landmark_dim', 2)
            self.num_neighbors   = getattr(args, 'num_neighbors', None)  # inferred at runtime if None
            self.neighbor_dim    = getattr(args, 'neighbor_dim', 4)
            self.num_agents      = getattr(args, 'num_agents', None)
            self.num_good_agents = getattr(args, 'num_good_agents', 2)
            self.num_adversaries = getattr(args, 'num_adversaries', 4)
            self.num_forests     = getattr(args, 'num_forests', 2)
            self.num_entities    = getattr(args, 'num_entities', None)

            if self._use_world_comm_hetero:
                self.num_landmarks = self.num_entities or getattr(args, 'num_landmarks', 1)
                self.num_food = max(self.num_landmarks - getattr(args, 'num_landmarks', 1) - self.num_forests, 0)
                expected_agent_state_dim = 4 + self.num_forests + self.num_good_agents
                legacy_agent_state_dim = 8
                if (
                    self.agent_state_dim not in (expected_agent_state_dim, legacy_agent_state_dim)
                    or self.landmark_dim != 4
                    or self.neighbor_dim != 5
                ):
                    raise ValueError(
                        "simple_world_comm HeteroGraph requires "
                        "--agent_state_dim {} (or legacy 8) --landmark_dim 4 --neighbor_dim 5".format(
                            expected_agent_state_dim
                        )
                    )
            elif self._use_tag_hetero:
                self.num_landmarks = getattr(args, 'num_landmarks', 2)
                if self.agent_state_dim != 4 or self.landmark_dim != 2 or self.neighbor_dim != 5:
                    raise ValueError(
                        "simple_tag HeteroGraph requires "
                        "--agent_state_dim 4 --landmark_dim 2 --neighbor_dim 5"
                    )
            else:
                self.num_landmarks = getattr(args, 'num_landmarks', 3)

            if self._use_parallel_ally_graph and self._use_entity_enemy_first_graph:
                raise ValueError("--use_parallel_ally_graph and --use_entity_enemy_first_graph are mutually exclusive")
            if self._use_parallel_ally_graph:
                graph_base_cls = ParallelAllyEntityGraphActorBase
            elif self._use_entity_enemy_first_graph:
                graph_base_cls = EntityEnemyFirstGraphActorBase
            else:
                graph_base_cls = HeteroGraphActorBase
            smd_kwargs = {}
            if graph_base_cls is HeteroGraphActorBase:
                smd_kwargs = dict(
                    use_smd=self._use_smd,
                    smd_candidate_top_k=getattr(args, 'smd_candidate_top_k', 3),
                    smd_student_hidden_dim=getattr(args, 'smd_student_hidden_dim', 128),
                    smd_edge_top_m=getattr(args, 'smd_edge_top_m', 1),
                    smd_edge_threshold=getattr(args, 'smd_edge_threshold', 0.5),
                    smd_use_topm_mask=getattr(args, 'smd_use_topm_mask', True),
                    use_smd_diffusion_teacher=getattr(args, 'use_smd_diffusion_teacher', True),
                    smd_num_diffusion_steps=getattr(args, 'smd_num_diffusion_steps', 50),
                    smd_beta_start=getattr(args, 'smd_beta_start', 1e-4),
                    smd_beta_end=getattr(args, 'smd_beta_end', 2e-2),
                    smd_teacher_hidden_dim=getattr(args, 'smd_teacher_hidden_dim', 128),
                    lambda_smd_diff=getattr(args, 'lambda_smd_diff', 0.01),
                    lambda_smd_distill=getattr(args, 'lambda_smd_distill', 0.05),
                    lambda_smd_sparse=getattr(args, 'lambda_smd_sparse', 0.001),
                    smd_target_degree=getattr(args, 'smd_target_degree', 1.0),
                    smd_pseudo_top_m=getattr(args, 'smd_pseudo_top_m', 1),
                    smd_use_positive_adv_only=getattr(args, 'smd_use_positive_adv_only', False),
                    smd_online_sampling=getattr(args, 'smd_online_sampling', False),
                    smd_debug_shapes=getattr(args, 'smd_debug_shapes', False),
                    use_gsd_bsd=self._use_gsd_bsd,
                    gsd_bsd_ally_candidate_k=getattr(args, 'gsd_bsd_ally_candidate_k', 4),
                    gsd_bsd_enemy_candidate_k=getattr(args, 'gsd_bsd_enemy_candidate_k', 4),
                    gsd_bsd_ally_edge_m=getattr(args, 'gsd_bsd_ally_edge_m', 2),
                    gsd_bsd_enemy_edge_m=getattr(args, 'gsd_bsd_enemy_edge_m', 1),
                    gsd_bsd_hidden_dim=getattr(args, 'gsd_bsd_hidden_dim', 64),
                    gsd_bsd_num_heads=getattr(args, 'gsd_bsd_num_heads', 2),
                    gsd_bsd_num_layers=getattr(args, 'gsd_bsd_num_layers', 1),
                    gsd_bsd_type_embedding_dim=getattr(args, 'gsd_bsd_type_embedding_dim', 8),
                    gsd_bsd_state_embedding_dim=getattr(args, 'gsd_bsd_state_embedding_dim', 8),
                    gsd_bsd_time_embedding_dim=getattr(args, 'gsd_bsd_time_embedding_dim', 16),
                    gsd_bsd_denoise_steps=getattr(args, 'gsd_bsd_denoise_steps', 1),
                    gsd_bsd_use_st_mask=getattr(args, 'gsd_bsd_use_st_mask', True),
                    gsd_bsd_enemy_base_score_source=getattr(args, 'gsd_bsd_enemy_base_score_source', 'auto'),
                    gsd_bsd_debug=getattr(args, 'gsd_bsd_debug', False),
                )
            elif self._use_gsd_bsd:
                raise RuntimeError("use_gsd_bsd=True but selected Actor branch does not support GSD-BSD")
            self.base = graph_base_cls(
                agent_state_dim=self.agent_state_dim,
                landmark_dim=self.landmark_dim,
                neighbor_dim=self.neighbor_dim,
                ally_dim=self.neighbor_dim,
                enemy_dim=self.neighbor_dim,
                ctx_dim=self.landmark_dim,
                hidden_size=self.hidden_size,
                use_orthogonal=self._use_orthogonal,
                use_ReLU=getattr(args, 'use_ReLU', True),
                use_gated_fusion=getattr(args, 'use_gated_fusion', False),
                **({} if self._use_parallel_ally_graph else dict(
                    k_max=getattr(args, 'k_max', 10),
                    top_k_filter=getattr(args, 'top_k_filter', 5),
                    use_knn=self._use_knn,
                    use_am_filter=self._use_am_filter,
                )),
                **smd_kwargs
            )

        # ============================================================
        # Mode B: ESMG (existing)
        # ============================================================
        elif self._use_esmg:
            self.self_dim = args.self_dim
            self.neighbor_dim = args.neighbor_dim
            self.k_max = getattr(args, 'k_max', 10)
            self.top_k_filter = getattr(args, 'top_k_filter', 5)
            self.num_agents = getattr(args, 'num_agents', None)
            self.scenario_name = getattr(args, 'scenario_name', None)
            self.num_graph_layers = getattr(args, 'esmg_layers', None) or getattr(args, 'num_graph_layers', 2)
            self.base = EgoSparseMultiplexGraph(
                self_dim=self.self_dim,
                neighbor_dim=self.neighbor_dim,
                hidden_size=self.hidden_size,
                k_max=self.k_max,
                top_k_filter=self.top_k_filter,
                num_layers=self.num_graph_layers,
                use_orthogonal=self._use_orthogonal,
                use_ReLU=getattr(args, 'use_ReLU', True),
                use_knn=self._use_knn,
                use_am_filter=self._use_am_filter,
            )
            belief_flat_dim = self.num_graph_layers * self.top_k_filter
            recon_target_dim = self.neighbor_dim * self.top_k_filter
            active_func = nn.ReLU() if getattr(args, 'use_ReLU', True) else nn.Tanh()
            self.recon_decoder = nn.Sequential(
                nn.Linear(belief_flat_dim, self.hidden_size),
                active_func,
                nn.Linear(self.hidden_size, recon_target_dim),
            )

        # ============================================================
        # Mode C: Vanilla MLP / CNN
        # ============================================================
        else:
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.algo = args.algorithm_name

    # ==================================================================
    # Observation parsers
    # ==================================================================
    def _pad_last_dim(self, x, target_dim):
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] > target_dim:
            return x[..., :target_dim]
        pad_shape = list(x.shape[:-1]) + [target_dim - x.shape[-1]]
        pad = x.new_zeros(*pad_shape)
        return torch.cat([x, pad], dim=-1)

    def _split_world_comm_hetero_obs(self, obs):
        """
        Adapt simple_world_comm observations to the generic HeteroGraph interface.

        Scenario layout:
          adversary obs:
            [p_vel(2), p_pos(2), entity_pos(E*2), other_pos((N-1)*2),
             good_vel(G*2), in_forest(F), prey_forest(G)]
          good-agent obs:
            [p_vel(2), p_pos(2), entity_pos(E*2), other_pos((N-1)*2),
             in_forest(F), other_good_vel((G-1)*2)]

        Returns:
          agent_state:  [B, 8]
          landmark_obs: [B, E, 4]  with [dx, dy, entity_type_0, entity_type_1]
          neighbor_obs: [B, N-1, 5] with [dx, dy, vx, vy, agent_type]
          landmark_mask: [B, E]
          neighbor_mask: [B, N-1], True means invalid padding
        """
        B = obs.shape[0]
        E = self.num_landmarks
        F = self.num_forests
        G = self.num_good_agents
        A = self.num_adversaries
        N_other = max((self.num_agents or (A + G)) - 1, 0)

        entity_dim = E * 2
        other_pos_dim = N_other * 2
        adv_obs_dim = 4 + entity_dim + other_pos_dim + G * 2 + F + G
        good_obs_dim = 4 + entity_dim + other_pos_dim + F + max(G - 1, 0) * 2

        if obs.shape[-1] not in (adv_obs_dim, good_obs_dim):
            raise ValueError(
                "simple_world_comm HeteroGraph obs dim {} does not match expected adversary dim {} "
                "or good-agent dim {} (num_agents={}, num_adversaries={}, num_good_agents={}, "
                "num_entities={}, num_forests={})".format(
                    obs.shape[-1], adv_obs_dim, good_obs_dim,
                    self.num_agents or (A + G), A, G, E, F,
                )
            )

        is_adversary_policy = obs.shape[-1] == adv_obs_dim

        idx = 0
        self_phys = obs[:, idx:idx + 4]
        idx += 4

        entity_pos = obs[:, idx:idx + entity_dim].contiguous().view(B, E, 2)
        idx += entity_dim

        other_pos = obs[:, idx:idx + other_pos_dim].contiguous().view(B, N_other, 2)
        idx += other_pos_dim

        base_landmarks = max(E - self.num_food - F, 0)
        type_codes = obs.new_zeros(E, 2)
        if base_landmarks > 0:
            type_codes[:base_landmarks, 0] = 1.0
        if self.num_food > 0:
            type_codes[base_landmarks:base_landmarks + self.num_food, 1] = 1.0
        if F > 0:
            type_codes[base_landmarks + self.num_food:, :] = -1.0
        entity_type = type_codes.unsqueeze(0).expand(B, -1, -1)
        landmark_obs = torch.cat([entity_pos, entity_type], dim=-1)
        landmark_mask = obs.new_ones(B, E)
        other_visible = (other_pos.abs().sum(dim=-1) > 1e-6).to(dtype=obs.dtype)

        if is_adversary_policy:
            good_vel = obs[:, idx:idx + G * 2].contiguous().view(B, G, 2)
            idx += G * 2
            in_forest = obs[:, idx:idx + F]
            idx += F
            prey_forest = obs[:, idx:idx + G]

            agent_context = self._pad_last_dim(
                torch.cat([in_forest, prey_forest], dim=-1),
                max(self.agent_state_dim - 4, 0),
            )
            agent_state = torch.cat([self_phys, agent_context], dim=-1)

            adv_count = max(A - 1, 0)
            good_count = G
            adv_pos = other_pos[:, :adv_count, :]
            good_pos = other_pos[:, adv_count:adv_count + good_count, :]

            adv_vel = obs.new_zeros(B, adv_count, 2)
            good_vel = self._pad_last_dim(good_vel, 2)
            adv_type = obs.new_ones(B, adv_count, 1)
            good_type = -obs.new_ones(B, good_count, 1)

            adv_neighbors = torch.cat([adv_pos, adv_vel, adv_type], dim=-1)
            good_neighbors = torch.cat([good_pos, good_vel, good_type], dim=-1)
            adv_mask = ~other_visible[:, :adv_count].to(dtype=torch.bool)
            good_mask = ~other_visible[:, adv_count:adv_count + good_count].to(dtype=torch.bool)
            ally_obs, enemy_obs = adv_neighbors, good_neighbors
            ally_mask, enemy_mask = adv_mask, good_mask
        else:
            in_forest = obs[:, idx:idx + F]
            idx += F
            other_good_count = max(G - 1, 0)
            other_good_vel = obs[:, idx:idx + other_good_count * 2].contiguous().view(B, other_good_count, 2)

            prey_forest = obs.new_zeros(B, G)
            agent_context = self._pad_last_dim(
                torch.cat([in_forest, prey_forest], dim=-1),
                max(self.agent_state_dim - 4, 0),
            )
            agent_state = torch.cat([self_phys, agent_context], dim=-1)

            adv_count = A
            good_count = other_good_count
            adv_pos = other_pos[:, :adv_count, :]
            good_pos = other_pos[:, adv_count:adv_count + good_count, :]

            adv_vel = obs.new_zeros(B, adv_count, 2)
            good_vel = self._pad_last_dim(other_good_vel, 2)
            adv_type = obs.new_ones(B, adv_count, 1)
            good_type = -obs.new_ones(B, good_count, 1)

            adv_neighbors = torch.cat([adv_pos, adv_vel, adv_type], dim=-1)
            good_neighbors = torch.cat([good_pos, good_vel, good_type], dim=-1)
            adv_mask = ~other_visible[:, :adv_count].to(dtype=torch.bool)
            good_mask = ~other_visible[:, adv_count:adv_count + good_count].to(dtype=torch.bool)
            ally_obs, enemy_obs = good_neighbors, adv_neighbors
            ally_mask, enemy_mask = good_mask, adv_mask

        return agent_state, ally_obs, enemy_obs, landmark_obs, ally_mask, enemy_mask, landmark_mask

    def _split_tag_hetero_obs(self, obs):
        """
        Adapt simple_tag observations to the generic HeteroGraph interface.

        Scenario layout:
          [p_vel(2), p_pos(2), landmark_pos(M*2), other_pos((N-1)*2),
           good_vel(G*2)]

        Only good-agent velocities are included by the environment. Missing
        adversary velocities are zero-filled.

        Returns:
          agent_state:  [B, 4]
          landmark_obs: [B, M, 2]
          neighbor_obs: [B, N-1, 5] with [dx, dy, vx, vy, agent_type]
        """
        B = obs.shape[0]
        M = self.num_landmarks
        G = self.num_good_agents
        A = self.num_adversaries
        N_other = max((self.num_agents or (A + G)) - 1, 0)

        landmark_dim = M * 2
        other_pos_dim = N_other * 2
        adv_obs_dim = 4 + landmark_dim + other_pos_dim + G * 2
        good_obs_dim = 4 + landmark_dim + other_pos_dim + max(G - 1, 0) * 2

        if obs.shape[-1] not in (adv_obs_dim, good_obs_dim):
            raise ValueError(
                "simple_tag HeteroGraph obs dim {} does not match expected adversary dim {} "
                "or good-agent dim {} (num_agents={}, num_adversaries={}, num_good_agents={}, "
                "num_landmarks={})".format(
                    obs.shape[-1], adv_obs_dim, good_obs_dim,
                    self.num_agents or (A + G), A, G, M,
                )
            )

        is_adversary_policy = obs.shape[-1] == adv_obs_dim

        idx = 0
        agent_state = obs[:, idx:idx + 4]
        idx += 4

        landmark_obs = obs[:, idx:idx + landmark_dim].contiguous().view(B, M, 2)
        idx += landmark_dim

        other_pos = obs[:, idx:idx + other_pos_dim].contiguous().view(B, N_other, 2)
        idx += other_pos_dim

        if is_adversary_policy:
            good_count = G
            good_vel = obs[:, idx:idx + good_count * 2].contiguous().view(B, good_count, 2)
            adv_count = max(A - 1, 0)
        else:
            good_count = max(G - 1, 0)
            good_vel = obs[:, idx:idx + good_count * 2].contiguous().view(B, good_count, 2)
            adv_count = A

        adv_pos = other_pos[:, :adv_count, :]
        good_pos = other_pos[:, adv_count:adv_count + good_count, :]

        adv_vel = obs.new_zeros(B, adv_count, 2)
        adv_type = obs.new_ones(B, adv_count, 1)
        good_type = -obs.new_ones(B, good_count, 1)

        adv_neighbors = torch.cat([adv_pos, adv_vel, adv_type], dim=-1)
        good_neighbors = torch.cat([good_pos, good_vel, good_type], dim=-1)

        landmark_mask = obs.new_ones(B, M)
        adv_mask = torch.zeros(B, adv_count, dtype=torch.bool, device=obs.device)
        good_mask = torch.zeros(B, good_count, dtype=torch.bool, device=obs.device)
        if is_adversary_policy:
            ally_obs, enemy_obs = adv_neighbors, good_neighbors
            ally_mask, enemy_mask = adv_mask, good_mask
        else:
            ally_obs, enemy_obs = good_neighbors, adv_neighbors
            ally_mask, enemy_mask = good_mask, adv_mask

        return agent_state, ally_obs, enemy_obs, landmark_obs, ally_mask, enemy_mask, landmark_mask

    def _split_hetero_obs(self, obs):
        """
        Unpack flat obs from original MAPPO simple_spread into three tensors.

        Original layout:
            [p_vel(2) | p_pos(2) | landmark_rel_pos(M*2) | other_pos(K*2) | comm(K*2)]

        Returns:
            agent_state:  [B, agent_state_dim]
            landmark_obs: [B, M, landmark_dim]
            neighbor_obs: [B, K, neighbor_dim]
        """
        B = obs.shape[0]
        idx = 0

        # agent physical state  (p_vel + p_pos)
        agent_state = obs[:, idx: idx + self.agent_state_dim]
        idx += self.agent_state_dim

        # landmark relative positions
        l_total = self.num_landmarks * self.landmark_dim
        landmark_flat = obs[:, idx: idx + l_total]
        landmark_obs = landmark_flat.contiguous().view(B, self.num_landmarks, self.landmark_dim)
        idx += l_total

        # remainder = neighbor features  (other_pos | comm)
        tail = obs[:, idx:]

        if tail.shape[-1] == 0:
            ally_obs = obs.new_zeros(B, 0, self.neighbor_dim)
            enemy_obs = obs.new_zeros(B, 0, self.neighbor_dim)
            ctx_mask = obs.new_ones(B, self.num_landmarks)
            ally_mask = torch.zeros(B, 0, dtype=torch.bool, device=obs.device)
            enemy_mask = torch.zeros(B, 0, dtype=torch.bool, device=obs.device)
            return agent_state, ally_obs, enemy_obs, landmark_obs, ally_mask, enemy_mask, ctx_mask

        # Determine number of neighbors
        if self.num_neighbors is not None:
            n_neigh = self.num_neighbors
        elif self.num_agents is not None:
            n_neigh = max(self.num_agents - 1, 0)
        else:
            n_neigh = tail.shape[-1] // self.neighbor_dim

        if n_neigh == 0:
            ally_obs = obs.new_zeros(B, 0, self.neighbor_dim)
            enemy_obs = obs.new_zeros(B, 0, self.neighbor_dim)
            ctx_mask = obs.new_ones(B, self.num_landmarks)
            ally_mask = torch.zeros(B, 0, dtype=torch.bool, device=obs.device)
            enemy_mask = torch.zeros(B, 0, dtype=torch.bool, device=obs.device)
            return agent_state, ally_obs, enemy_obs, landmark_obs, ally_mask, enemy_mask, ctx_mask

        # Original MPE: other_pos(K*2) then comm(K*2) — interleave per neighbor
        if self.neighbor_dim == 4 and tail.shape[-1] == n_neigh * 4:
            pos_end = n_neigh * 2
            other_pos = tail[:, :pos_end].contiguous().view(B, n_neigh, 2)
            comm      = tail[:, pos_end: pos_end + n_neigh * 2].contiguous().view(B, n_neigh, 2)
            neighbor_obs = torch.cat([other_pos, comm], dim=-1)
        else:
            neighbor_obs = tail.contiguous().view(B, n_neigh, self.neighbor_dim)

        ally_obs = neighbor_obs
        enemy_obs = obs.new_zeros(B, 0, self.neighbor_dim)
        ctx_mask = obs.new_ones(B, self.num_landmarks)
        ally_mask = torch.zeros(B, ally_obs.shape[1], dtype=torch.bool, device=obs.device)
        enemy_mask = torch.zeros(B, 0, dtype=torch.bool, device=obs.device)
        return agent_state, ally_obs, enemy_obs, landmark_obs, ally_mask, enemy_mask, ctx_mask

    def _split_esmg_obs(self, obs):
        """
        Parse original MPE simple_spread observation layout into ESMG inputs.

        Original MAPPO simple_spread layout:
            [self_obs | other_pos(flat) | comm(flat)]
        where:
            self_obs = p_vel(2) + p_pos(2) + landmark_rel_pos(2 * num_landmarks)
            other_pos(flat) = (num_agents - 1) * 2
            comm(flat) = (num_agents - 1) * 2

        Returns:
            self_obs: [batch, self_dim]
            neighbor_obs: [batch, num_neighbors, neighbor_dim]
        """
        batch_size = obs.shape[0]
        self_obs = obs[:, :self.self_dim]
        tail = obs[:, self.self_dim:]

        if tail.shape[-1] == 0:
            neighbor_obs = obs.new_zeros((batch_size, 0, self.neighbor_dim))
            return self_obs, neighbor_obs

        if self.neighbor_dim != 4:
            if tail.shape[-1] % self.neighbor_dim != 0:
                raise ValueError(
                    "ESMG fallback parsing requires obs tail divisible by neighbor_dim, got tail={} neighbor_dim={}".format(
                        tail.shape[-1], self.neighbor_dim
                    )
                )
            num_neighbors = tail.shape[-1] // self.neighbor_dim
            neighbor_obs = tail.contiguous().view(batch_size, num_neighbors, self.neighbor_dim)
            return self_obs, neighbor_obs

        if tail.shape[-1] % 2 != 0:
            raise ValueError("Invalid MPE observation tail length {} for neighbor extraction".format(tail.shape[-1]))

        if self.num_agents is not None:
            expected_neighbors = max(self.num_agents - 1, 0)
            expected_tail = expected_neighbors * self.neighbor_dim
            if tail.shape[-1] != expected_tail:
                raise ValueError(
                    "Observation tail length {} does not match num_agents-derived expectation {} (num_agents={}, neighbor_dim={})".format(
                        tail.shape[-1], expected_tail, self.num_agents, self.neighbor_dim
                    )
                )
            num_neighbors = expected_neighbors
        else:
            num_neighbors = tail.shape[-1] // self.neighbor_dim

        if num_neighbors == 0:
            neighbor_obs = obs.new_zeros((batch_size, 0, self.neighbor_dim))
            return self_obs, neighbor_obs

        pos_dim = num_neighbors * 2
        comm_dim = num_neighbors * 2
        if tail.shape[-1] != pos_dim + comm_dim:
            raise ValueError(
                "Cannot parse original MPE obs tail of length {} into other_pos + comm for {} neighbors".format(
                    tail.shape[-1], num_neighbors
                )
            )

        other_pos = tail[:, :pos_dim].contiguous().view(batch_size, num_neighbors, 2)
        comm = tail[:, pos_dim:pos_dim + comm_dim].contiguous().view(batch_size, num_neighbors, 2)
        neighbor_obs = torch.cat([other_pos, comm], dim=-1)
        return self_obs, neighbor_obs

    # ==================================================================
    # Forward helpers (shared feature extraction per mode)
    # ==================================================================
    def _extract_features(self, obs):
        """Return (actor_features, latent_beliefs).  latent_beliefs may be None."""
        if self._use_hetero_graph:
            if self._use_world_comm_hetero:
                agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask = self._split_world_comm_hetero_obs(obs)
                graph_out = self.base(
                    agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask
                )
            elif self._use_tag_hetero:
                agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask = self._split_tag_hetero_obs(obs)
                graph_out = self.base(
                    agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask
                )
            else:
                agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask = self._split_hetero_obs(obs)
                graph_out = self.base(
                    agent_state, ally_obs, enemy_obs, ctx_obs, ally_mask, enemy_mask, ctx_mask
                )
            if len(graph_out) == 3:
                actor_features, latent_beliefs, graph_gates = graph_out
                self.last_graph_gates = graph_gates
            else:
                actor_features, latent_beliefs = graph_out
                self.last_graph_gates = None
            return actor_features, latent_beliefs
        elif self._use_esmg:
            self_obs, neighbor_obs = self._split_esmg_obs(obs)
            return self.base(self_obs, neighbor_obs)
        else:
            return self.base(obs), None

    # ==================================================================
    # Public methods
    # ==================================================================
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.

        :return actions, action_log_probs, rnn_states, latent_beliefs
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, latent_beliefs = self._extract_features(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states, latent_beliefs

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, _ = self._extract_features(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.algo == "hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions_trpo(
                actor_features, action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None)
            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                actor_features, action, available_actions,
                active_masks=active_masks if self._use_policy_active_masks else None)

        return action_log_probs, dist_entropy

    def evaluate_actions_with_smd(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate actions and return SMD auxiliary tensors produced by Hop3.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, _ = self._extract_features(obs)
        smd_aux = getattr(self.base, "last_smd_aux", {}) if self._use_hetero_graph else {}

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None)

        return action_log_probs, dist_entropy, smd_aux

    def evaluate_actions_with_gsd_bsd(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate actions and return GSD-BSD auxiliary tensors produced by Hop3.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features, _ = self._extract_features(obs)
        bsd_aux = getattr(self.base, "last_gsd_bsd_aux", {}) if self._use_hetero_graph else {}

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None)

        return action_log_probs, dist_entropy, bsd_aux

    def evaluate_actions_with_recon(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Like evaluate_actions, but also returns reconstruction loss for ESMG AM filter training.
        For HeteroGraph mode, recon_loss is always 0.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        recon_loss = torch.tensor(0.0, device=obs.device)

        if self._use_hetero_graph:
            actor_features, latent_beliefs = self._extract_features(obs)
            # HeteroGraph does not use reconstruction loss (yet)

        elif self._use_esmg:
            batch_size = obs.shape[0]
            self_obs, neighbor_obs = self._split_esmg_obs(obs)
            actor_features, latent_beliefs = self.base(self_obs, neighbor_obs)

            # --- Reconstruction loss (only when AM filter is enabled) ---
            if self._use_am_filter:
                beliefs_flat = latent_beliefs.reshape(batch_size, -1)

                expected_flat_dim = self.num_graph_layers * self.top_k_filter
                if beliefs_flat.shape[-1] < expected_flat_dim:
                    pad = torch.zeros(batch_size, expected_flat_dim - beliefs_flat.shape[-1],
                                      device=obs.device, dtype=obs.dtype)
                    beliefs_flat = torch.cat([beliefs_flat, pad], dim=-1)
                elif beliefs_flat.shape[-1] > expected_flat_dim:
                    beliefs_flat = beliefs_flat[:, :expected_flat_dim]

                recon_pred = self.recon_decoder(beliefs_flat)

                num_neighbors = neighbor_obs.shape[1]
                effective_k_max = min(self.k_max, num_neighbors) if self._use_knn else num_neighbors
                effective_top_k = min(self.top_k_filter, effective_k_max)

                if self._use_knn and effective_k_max > 0:
                    dx = neighbor_obs[:, :, 0]
                    dy = neighbor_obs[:, :, 1]
                    dist_sq = dx ** 2 + dy ** 2
                    _, knn_indices = torch.topk(dist_sq, k=effective_k_max, dim=-1, largest=False)
                    knn_indices_exp = knn_indices.unsqueeze(-1).expand(-1, -1, self.neighbor_dim)
                    knn_obs = torch.gather(neighbor_obs, dim=1, index=knn_indices_exp)
                else:
                    knn_obs = neighbor_obs

                if effective_top_k > 0:
                    neighbor_h = self.base.neighbor_encoder(knn_obs)
                    w = self.base.am_filter(neighbor_h).squeeze(-1)
                    _, filter_idx = torch.topk(w, k=effective_top_k, dim=-1, largest=True)

                    filter_idx_obs = filter_idx.unsqueeze(-1).expand(-1, -1, self.neighbor_dim)
                    filtered_obs = torch.gather(knn_obs, dim=1, index=filter_idx_obs)

                    w_filtered = torch.gather(w, dim=1, index=filter_idx).unsqueeze(-1)
                    weighted_obs = (w_filtered * filtered_obs).reshape(batch_size, -1)
                else:
                    weighted_obs = torch.zeros(batch_size, 0, device=obs.device, dtype=obs.dtype)

                target_dim = self.neighbor_dim * self.top_k_filter
                if weighted_obs.shape[-1] < target_dim:
                    pad = torch.zeros(batch_size, target_dim - weighted_obs.shape[-1],
                                      device=obs.device, dtype=obs.dtype)
                    weighted_obs = torch.cat([weighted_obs, pad], dim=-1)
                elif weighted_obs.shape[-1] > target_dim:
                    weighted_obs = weighted_obs[:, :target_dim]

                recon_loss = F.mse_loss(recon_pred, weighted_obs.detach())
        else:
            actor_features, _ = self._extract_features(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None)

        return action_log_probs, dist_entropy, recon_loss


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
