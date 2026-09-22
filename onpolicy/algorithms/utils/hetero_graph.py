"""
Entity-first Heterogeneous Graph Actor Base.

Serial reasoning path:
  Hop 1  L2A_Base     - ego query attends over environment entities.
  Hop 2  A2A_Coord    - ego + entity context attends over neighbour agents.
  Hop 3  A2A_Refined  - ego + peer intent independently gates all neighbours.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Scaled dot-product multi-head attention with optional additive mask."""

    def __init__(self, query_dim, key_dim, hidden_size, num_heads=1, use_orthogonal=True, use_ReLU=True):
        super(GraphAttentionLayer, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(query_dim, hidden_size)
        self.k_proj = nn.Linear(key_dim, hidden_size)
        self.v_proj = nn.Linear(key_dim, hidden_size)
        self._init_weights(use_orthogonal, use_ReLU)

    def _init_weights(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _valid_to_attention_mask(valid_mask, dtype):
        if valid_mask is None:
            return None
        keep = valid_mask.to(dtype=torch.bool)                              # Shape: [B, S]
        return torch.zeros_like(valid_mask, dtype=dtype).masked_fill(~keep, float('-inf'))  # Shape: [B, S]

    def forward(self, query, key_value, valid_mask=None):
        """
        Args:
            query:      [B, Dq]
            key_value:  [B, S, Dk]
            valid_mask: [B, S] or None. 1/True means valid.

        Returns:
            msg:        [B, H]
            attn_mean:  [B, S]
        """
        B = query.shape[0]
        S = key_value.shape[1]
        if S == 0:
            msg = query.new_zeros(B, self.hidden_size)                      # Shape: [B, H]
            attn = query.new_zeros(B, 1)                                    # Shape: [B, 1]
            return msg, attn

        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim)     # Shape: [B, 1, Nh, Dh]
        K = self.k_proj(key_value).view(B, S, self.num_heads, self.head_dim) # Shape: [B, S, Nh, Dh]
        V = self.v_proj(key_value).view(B, S, self.num_heads, self.head_dim) # Shape: [B, S, Nh, Dh]

        Q = Q.transpose(1, 2)                                                # Shape: [B, Nh, 1, Dh]
        K = K.transpose(1, 2)                                                # Shape: [B, Nh, S, Dh]
        V = V.transpose(1, 2)                                                # Shape: [B, Nh, S, Dh]

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(float(self.head_dim))  # Shape: [B, Nh, 1, S]
        attn_mask = self._valid_to_attention_mask(valid_mask, query.dtype)   # Shape: [B, S] or None
        if attn_mask is not None:
            logits = logits + attn_mask[:, None, None, :]                    # Shape: [B, Nh, 1, S]
            all_pruned = torch.isneginf(logits).all(dim=-1, keepdim=True)    # Shape: [B, Nh, 1, 1]
            logits = logits.masked_fill(all_pruned, 0.0)                     # Shape: [B, Nh, 1, S]

        attn = F.softmax(logits, dim=-1)                                     # Shape: [B, Nh, 1, S]
        if attn_mask is not None:
            keep = (~torch.isneginf(attn_mask)).to(dtype=attn.dtype)         # Shape: [B, S]
            attn = attn * keep[:, None, None, :]                             # Shape: [B, Nh, 1, S]
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)          # Shape: [B, Nh, 1, 1]
            attn = attn / denom                                              # Shape: [B, Nh, 1, S]

        out = torch.matmul(attn, V)                                          # Shape: [B, Nh, 1, Dh]
        out = out.transpose(1, 2).contiguous().view(B, self.hidden_size)      # Shape: [B, H]

        if valid_mask is not None:
            has_valid = valid_mask.to(dtype=torch.bool).any(dim=-1, keepdim=True)  # Shape: [B, 1]
            out = out * has_valid.to(dtype=query.dtype)                      # Shape: [B, H]

        attn_mean = attn.squeeze(2).mean(dim=1)                              # Shape: [B, S]
        return out, attn_mean


class GatedConcatenationFusion(nn.Module):
    """
    Feature-level gated concatenation.

    This module keeps intent/refined channels separate. It generates a
    non-zero-sum sigmoid gate for each feature channel, gates each branch
    independently, then concatenates the gated features before projection.
    """

    def __init__(self, dim_intent, dim_refined, hidden_size, use_orthogonal=True, use_ReLU=True):
        super(GatedConcatenationFusion, self).__init__()
        self.dim_intent = dim_intent
        self.dim_refined = dim_refined
        self.raw_dim = dim_intent + dim_refined

        act_fn = nn.ReLU() if use_ReLU else nn.Tanh()
        self.gate = nn.Linear(self.raw_dim, self.raw_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_size),
            act_fn,
            nn.LayerNorm(hidden_size),
        )
        self._init_weights(use_orthogonal, use_ReLU)

    def _init_weights(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, m_intent, m_refined):
        X_raw = torch.cat([m_intent, m_refined], dim=-1)              # Shape: [B, dim_intent+dim_refined]
        G = torch.sigmoid(self.gate(X_raw))                           # Shape: [B, dim_intent+dim_refined]
        g_intent, g_refined = torch.split(
            G, [self.dim_intent, self.dim_refined], dim=-1
        )                                                             # Shape: [B, dim_intent], [B, dim_refined]
        m_intent_gated = m_intent * g_intent                          # Shape: [B, dim_intent]
        m_refined_gated = m_refined * g_refined                       # Shape: [B, dim_refined]
        X_gated = torch.cat([m_intent_gated, m_refined_gated], dim=-1) # Shape: [B, dim_intent+dim_refined]
        f_out = self.proj(X_gated)                                    # Shape: [B, hidden_size]
        return f_out, g_intent, g_refined


class ParallelAllyEntityGraphActorBase(nn.Module):
    """
    Parallel two-branch graph feature extractor.

    Branch 1 attends from ego state to all ally agents.
    Branch 2 attends from ego state to all other nodes: ally agents,
    enemy agents, and context entities. Agent/entity raw dimensions may differ,
    so branch 2 projects each node family before concatenating on sequence dim.
    """

    def __init__(
        self,
        agent_state_dim,
        landmark_dim,
        neighbor_dim,
        hidden_size=64,
        use_orthogonal=True,
        use_ReLU=True,
        use_gated_fusion=False,
        ally_dim=None,
        enemy_dim=None,
        ctx_dim=None,
    ):
        super(ParallelAllyEntityGraphActorBase, self).__init__()
        self.agent_state_dim = agent_state_dim
        self.ally_dim = ally_dim or neighbor_dim
        self.enemy_dim = enemy_dim or neighbor_dim
        self.ctx_dim = ctx_dim or landmark_dim
        self.hidden_size = hidden_size
        self.use_gated_fusion = use_gated_fusion
        self.last_g_intent = None
        self.last_g_refined = None

        H = hidden_size
        act_fn = nn.ReLU() if use_ReLU else nn.Tanh()

        self.ally_branch = GraphAttentionLayer(
            query_dim=agent_state_dim,
            key_dim=self.ally_dim,
            hidden_size=H,
            num_heads=1,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

        self.num_heads_all = 1
        self.head_dim_all = H // self.num_heads_all
        self.q_all = nn.Linear(agent_state_dim, H)
        self.k_all_ally = nn.Linear(self.ally_dim, H)
        self.v_all_ally = nn.Linear(self.ally_dim, H)
        self.k_all_enemy = nn.Linear(self.enemy_dim, H)
        self.v_all_enemy = nn.Linear(self.enemy_dim, H)
        self.k_all_ctx = nn.Linear(self.ctx_dim, H)
        self.v_all_ctx = nn.Linear(self.ctx_dim, H)
        self.out_all = nn.Linear(H, H)
        self._init_parallel_layers(use_orthogonal, use_ReLU)

        if self.use_gated_fusion:
            self.fusion = GatedConcatenationFusion(
                dim_intent=H,
                dim_refined=H,
                hidden_size=H,
                use_orthogonal=use_orthogonal,
                use_ReLU=use_ReLU,
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(H * 2, H),
                act_fn,
                nn.LayerNorm(H),
            )
            self._init_fusion(use_orthogonal, use_ReLU)

    def _init_parallel_layers(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in (
            self.q_all, self.k_all_ally, self.v_all_ally,
            self.k_all_enemy, self.v_all_enemy, self.k_all_ctx,
            self.v_all_ctx, self.out_all,
        ):
            if use_orthogonal:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _init_fusion(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _mask_to_valid(mask):
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            return ~mask
        return mask > 0

    def _project_nodes(self, x, k_layer, v_layer, B, N):
        K = k_layer(x).view(B, N, self.num_heads_all, self.head_dim_all).transpose(1, 2)  # Shape: [B, heads, N, head_dim]
        V = v_layer(x).view(B, N, self.num_heads_all, self.head_dim_all).transpose(1, 2)  # Shape: [B, heads, N, head_dim]
        return K, V

    def _masked_attention(self, Q, K, V, valid_mask):
        B = Q.shape[0]
        S = K.shape[-2]
        if S == 0:
            return Q.new_zeros(B, self.hidden_size), Q.new_zeros(B, 0)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(float(self.head_dim_all))  # Shape: [B, heads, 1, S]
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=Q.device, dtype=torch.bool)      # Shape: [B, S]
            logits = logits.masked_fill(~valid_mask[:, None, None, :], -1e9)   # Shape: [B, heads, 1, S]
            no_valid = ~valid_mask.any(dim=-1, keepdim=True)                  # Shape: [B, 1]
            logits = logits.masked_fill(no_valid[:, None, :, None], 0.0)      # Shape: [B, heads, 1, S]

        attn = F.softmax(logits, dim=-1)                                      # Shape: [B, heads, 1, S]
        if valid_mask is not None:
            keep = valid_mask.to(dtype=attn.dtype)                            # Shape: [B, S]
            attn = attn * keep[:, None, None, :]                              # Shape: [B, heads, 1, S]
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)     # Shape: [B, heads, 1, S]

        out = torch.matmul(attn, V)                                           # Shape: [B, heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, self.hidden_size)       # Shape: [B, H]
        if valid_mask is not None:
            out = out * valid_mask.any(dim=-1, keepdim=True).to(dtype=out.dtype)  # Shape: [B, H]
        return out, attn.squeeze(2).mean(dim=1)                               # Shape: [B, H], [B, S]

    def forward(self, agent_state, ally_obs, enemy_obs, ctx_obs,
                ally_mask=None, enemy_mask=None, ctx_mask=None):
        B = agent_state.shape[0]
        ally_valid = self._mask_to_valid(ally_mask)
        enemy_valid = self._mask_to_valid(enemy_mask)
        ctx_valid = self._mask_to_valid(ctx_mask)

        msg_ally, attn_ally = self.ally_branch(
            agent_state, ally_obs, valid_mask=ally_valid
        )                                                                      # Shape: [B, H], [B, N_ally]

        N_ally = ally_obs.shape[1]
        N_enemy = enemy_obs.shape[1]
        N_ctx = ctx_obs.shape[1]
        Q_all = self.q_all(agent_state).view(
            B, 1, self.num_heads_all, self.head_dim_all
        ).transpose(1, 2)                                                      # Shape: [B, heads, 1, head_dim]
        K_ally, V_ally = self._project_nodes(ally_obs, self.k_all_ally, self.v_all_ally, B, N_ally)
        K_enemy, V_enemy = self._project_nodes(enemy_obs, self.k_all_enemy, self.v_all_enemy, B, N_enemy)
        K_ctx, V_ctx = self._project_nodes(ctx_obs, self.k_all_ctx, self.v_all_ctx, B, N_ctx)
        K_all = torch.cat([K_ally, K_enemy, K_ctx], dim=-2)                    # Shape: [B, heads, N_ally+N_enemy+N_ctx, head_dim]
        V_all = torch.cat([V_ally, V_enemy, V_ctx], dim=-2)                    # Shape: [B, heads, N_ally+N_enemy+N_ctx, head_dim]

        if ally_valid is None:
            ally_valid = agent_state.new_ones(B, N_ally, dtype=torch.bool)
        if enemy_valid is None:
            enemy_valid = agent_state.new_ones(B, N_enemy, dtype=torch.bool)
        if ctx_valid is None:
            ctx_valid = agent_state.new_ones(B, N_ctx, dtype=torch.bool)
        all_valid = torch.cat([ally_valid, enemy_valid, ctx_valid], dim=-1)    # Shape: [B, N_ally+N_enemy+N_ctx]
        msg_all_raw, attn_all = self._masked_attention(Q_all, K_all, V_all, all_valid)
        msg_all = self.out_all(msg_all_raw)                                   # Shape: [B, H]

        if self.use_gated_fusion:
            features, g_ally, g_all = self.fusion(msg_ally, msg_all)          # Shape: [B, H], [B, H], [B, H]
            self.last_g_intent = g_ally
            self.last_g_refined = g_all
        else:
            fused = torch.cat([msg_ally, msg_all], dim=-1)                    # Shape: [B, 2H]
            features = self.fusion_mlp(fused)                                 # Shape: [B, H]
            self.last_g_intent = None
            self.last_g_refined = None

        max_pool = max(attn_ally.shape[1], attn_all.shape[1])

        def _pad(t, target):
            if t.shape[1] < target:
                return F.pad(t, (0, target - t.shape[1]))                     # Shape: [B, target]
            return t

        latent_beliefs = torch.stack(
            [_pad(attn_ally, max_pool), _pad(attn_all, max_pool)], dim=1
        )                                                                      # Shape: [B, 2, max_pool]

        if self.use_gated_fusion:
            return features, latent_beliefs, (g_ally, g_all)
        return features, latent_beliefs

    def get_last_gates(self):
        return self.last_g_intent, self.last_g_refined


class EntityEnemyFirstGraphActorBase(nn.Module):
    """
    Entity/enemy -> ally -> entity/enemy serial graph feature extractor.

    Hop1 attends over target nodes, where target nodes are enemy agents plus
    context entities. Hop2 attends over ally agents. Hop3 returns to the same
    target node family after optional KNN and AM/top-k filtering.
    """

    def __init__(
        self,
        agent_state_dim,
        landmark_dim,
        neighbor_dim,
        hidden_size=64,
        k_max=10,
        top_k_filter=5,
        use_knn=True,
        use_am_filter=True,
        use_orthogonal=True,
        use_ReLU=True,
        use_gated_fusion=False,
        ally_dim=None,
        enemy_dim=None,
        ctx_dim=None,
    ):
        super(EntityEnemyFirstGraphActorBase, self).__init__()
        self.agent_state_dim = agent_state_dim
        self.ally_dim = ally_dim or neighbor_dim
        self.enemy_dim = enemy_dim or neighbor_dim
        self.ctx_dim = ctx_dim or landmark_dim
        self.hidden_size = hidden_size
        self.k_max = k_max
        self.top_k_filter = top_k_filter
        self.use_knn = use_knn
        self.use_am_filter = use_am_filter
        self.use_gated_fusion = use_gated_fusion
        self.num_heads = 1
        self.head_dim = hidden_size // self.num_heads
        self.last_g_intent = None
        self.last_g_refined = None

        H = hidden_size
        act_fn = nn.ReLU() if use_ReLU else nn.Tanh()

        self.q_target_base = nn.Linear(agent_state_dim, H)
        self.k_enemy = nn.Linear(self.enemy_dim, H)
        self.v_enemy = nn.Linear(self.enemy_dim, H)
        self.k_ctx = nn.Linear(self.ctx_dim, H)
        self.v_ctx = nn.Linear(self.ctx_dim, H)
        self.score_enemy = nn.Linear(self.enemy_dim, H)
        self.score_ctx = nn.Linear(self.ctx_dim, H)
        self.out_target_base = nn.Linear(H, H)

        self.ally_coord = GraphAttentionLayer(
            query_dim=agent_state_dim + H,
            key_dim=self.ally_dim,
            hidden_size=H,
            num_heads=1,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

        self.synergy_mlp = nn.Sequential(
            nn.Linear(H + H, H),
            act_fn,
            nn.Linear(H, 1),
        )
        self.q_target_refined = nn.Linear(agent_state_dim + H, H)
        self.out_target_refined = nn.Linear(H, H)
        self._init_manual_layers(use_orthogonal, use_ReLU)

        if self.use_gated_fusion:
            self.fusion = GatedConcatenationFusion(
                dim_intent=H,
                dim_refined=H,
                hidden_size=H,
                use_orthogonal=use_orthogonal,
                use_ReLU=use_ReLU,
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(H * 2, H),
                act_fn,
                nn.LayerNorm(H),
            )
            self._init_fusion(use_orthogonal, use_ReLU)

    def _init_manual_layers(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        modules = [
            self.q_target_base, self.k_enemy, self.v_enemy, self.k_ctx,
            self.v_ctx, self.score_enemy, self.score_ctx, self.out_target_base,
            self.q_target_refined, self.out_target_refined,
        ]
        modules.extend([m for m in self.synergy_mlp.modules() if isinstance(m, nn.Linear)])
        for m in modules:
            if use_orthogonal:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _init_fusion(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _mask_to_valid(mask):
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            return ~mask
        return mask > 0

    def _to_heads(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)    # Shape: [B, heads, S, head_dim]

    def _masked_attention(self, q, k_flat, v_flat, valid_mask):
        B = q.shape[0]
        S = k_flat.shape[1]
        if S == 0:
            return q.new_zeros(B, self.hidden_size), q.new_zeros(B, 0)

        Q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)       # Shape: [B, heads, 1, head_dim]
        K = self._to_heads(k_flat)                                            # Shape: [B, heads, S, head_dim]
        V = self._to_heads(v_flat)                                            # Shape: [B, heads, S, head_dim]
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(float(self.head_dim))  # Shape: [B, heads, 1, S]
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=q.device, dtype=torch.bool)     # Shape: [B, S]
            logits = logits.masked_fill(~valid_mask[:, None, None, :], -1e9)  # Shape: [B, heads, 1, S]
            no_valid = ~valid_mask.any(dim=-1, keepdim=True)                 # Shape: [B, 1]
            logits = logits.masked_fill(no_valid[:, None, :, None], 0.0)     # Shape: [B, heads, 1, S]

        attn = F.softmax(logits, dim=-1)                                      # Shape: [B, heads, 1, S]
        if valid_mask is not None:
            keep = valid_mask.to(dtype=attn.dtype)                            # Shape: [B, S]
            attn = attn * keep[:, None, None, :]                              # Shape: [B, heads, 1, S]
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)     # Shape: [B, heads, 1, S]

        out = torch.matmul(attn, V)                                           # Shape: [B, heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, self.hidden_size)       # Shape: [B, H]
        if valid_mask is not None:
            out = out * valid_mask.any(dim=-1, keepdim=True).to(dtype=out.dtype)  # Shape: [B, H]
        return out, attn.squeeze(2).mean(dim=1)                               # Shape: [B, H], [B, S]

    def _build_target_nodes(self, enemy_obs, ctx_obs, enemy_valid, ctx_valid):
        B = enemy_obs.shape[0]
        N_enemy = enemy_obs.shape[1]
        N_ctx = ctx_obs.shape[1]
        k_enemy = self.k_enemy(enemy_obs)                                      # Shape: [B, N_enemy, H]
        v_enemy = self.v_enemy(enemy_obs)                                      # Shape: [B, N_enemy, H]
        k_ctx = self.k_ctx(ctx_obs)                                            # Shape: [B, N_ctx, H]
        v_ctx = self.v_ctx(ctx_obs)                                            # Shape: [B, N_ctx, H]
        score_enemy = self.score_enemy(enemy_obs)                              # Shape: [B, N_enemy, H]
        score_ctx = self.score_ctx(ctx_obs)                                    # Shape: [B, N_ctx, H]
        k_all = torch.cat([k_enemy, k_ctx], dim=1)                             # Shape: [B, N_enemy+N_ctx, H]
        v_all = torch.cat([v_enemy, v_ctx], dim=1)                             # Shape: [B, N_enemy+N_ctx, H]
        score_all = torch.cat([score_enemy, score_ctx], dim=1)                 # Shape: [B, N_enemy+N_ctx, H]

        if enemy_valid is None:
            enemy_valid = enemy_obs.new_ones(B, N_enemy, dtype=torch.bool)
        if ctx_valid is None:
            ctx_valid = ctx_obs.new_ones(B, N_ctx, dtype=torch.bool)
        target_valid = torch.cat([enemy_valid, ctx_valid], dim=1)              # Shape: [B, N_enemy+N_ctx]

        enemy_xy = enemy_obs[:, :, :2] if N_enemy > 0 else enemy_obs.new_zeros(B, 0, 2)
        ctx_xy = ctx_obs[:, :, :2] if N_ctx > 0 else ctx_obs.new_zeros(B, 0, 2)
        target_xy = torch.cat([enemy_xy, ctx_xy], dim=1)                       # Shape: [B, N_enemy+N_ctx, 2]
        return k_all, v_all, score_all, target_xy, target_valid

    def forward(self, agent_state, ally_obs, enemy_obs, ctx_obs,
                ally_mask=None, enemy_mask=None, ctx_mask=None):
        B = agent_state.shape[0]
        H = self.hidden_size
        ally_valid = self._mask_to_valid(ally_mask)
        enemy_valid = self._mask_to_valid(enemy_mask)
        ctx_valid = self._mask_to_valid(ctx_mask)
        target_k, target_v, target_score, target_xy, target_valid = self._build_target_nodes(
            enemy_obs, ctx_obs, enemy_valid, ctx_valid
        )

        # Hop1: ego attends over entities + enemies.
        q_target_base = self.q_target_base(agent_state)                        # Shape: [B, H]
        m_target_base_raw, _attn_target_base = self._masked_attention(
            q_target_base, target_k, target_v, target_valid
        )                                                                      # Shape: [B, H], [B, N_enemy+N_ctx]
        m_target_base = self.out_target_base(m_target_base_raw)                # Shape: [B, H]

        # Hop2: ego + target context attends over allies.
        q_ally = torch.cat([agent_state, m_target_base], dim=-1)               # Shape: [B, d_agent+H]
        m_ally, attn_ally = self.ally_coord(
            q_ally, ally_obs, valid_mask=ally_valid
        )                                                                      # Shape: [B, H], [B, N_ally]

        # Hop3: ego + ally intent returns to filtered entities + enemies.
        S = target_k.shape[1]
        if S == 0:
            m_target_refined = agent_state.new_zeros(B, H)                    # Shape: [B, H]
            attn_target_refined = agent_state.new_zeros(B, 0)                 # Shape: [B, 0]
        else:
            dist_sq = target_xy[:, :, 0] ** 2 + target_xy[:, :, 1] ** 2       # Shape: [B, N_enemy+N_ctx]
            dist_sq = dist_sq.masked_fill(~target_valid, float('inf'))         # Shape: [B, N_enemy+N_ctx]
            effective_k = min(self.k_max if self.use_knn else S, S)
            _, knn_idx = torch.topk(
                dist_sq, k=effective_k, dim=-1, largest=False
            )                                                                  # Shape: [B, k_max]
            knn_k = torch.gather(target_k, 1, knn_idx.unsqueeze(-1).expand(-1, -1, H))  # Shape: [B, k_max, H]
            knn_v = torch.gather(target_v, 1, knn_idx.unsqueeze(-1).expand(-1, -1, H))  # Shape: [B, k_max, H]
            knn_score = torch.gather(target_score, 1, knn_idx.unsqueeze(-1).expand(-1, -1, H))  # Shape: [B, k_max, H]
            knn_valid = torch.gather(target_valid, 1, knn_idx)                # Shape: [B, k_max]

            if self.use_am_filter:
                ally_expand = m_ally.unsqueeze(1).expand(-1, effective_k, -1) # Shape: [B, k_max, H]
                score_in = torch.cat([ally_expand, knn_score], dim=-1)         # Shape: [B, k_max, 2H]
                scores = self.synergy_mlp(score_in).squeeze(-1)               # Shape: [B, k_max]
                scores = scores.masked_fill(~knn_valid, -1e9)                 # Shape: [B, k_max]
                filter_k = min(self.top_k_filter, effective_k)
                _, filter_idx = torch.topk(
                    scores, k=filter_k, dim=-1, largest=True
                )                                                              # Shape: [B, k_filter]
                filtered_k = torch.gather(knn_k, 1, filter_idx.unsqueeze(-1).expand(-1, -1, H))  # Shape: [B, k_filter, H]
                filtered_v = torch.gather(knn_v, 1, filter_idx.unsqueeze(-1).expand(-1, -1, H))  # Shape: [B, k_filter, H]
                filtered_valid = torch.gather(knn_valid, 1, filter_idx)        # Shape: [B, k_filter]
            else:
                filtered_k = knn_k                                            # Shape: [B, k_max, H]
                filtered_v = knn_v                                            # Shape: [B, k_max, H]
                filtered_valid = knn_valid                                    # Shape: [B, k_max]

            q_target_refined = self.q_target_refined(
                torch.cat([agent_state, m_ally], dim=-1)
            )                                                                  # Shape: [B, H]
            m_target_refined_raw, attn_target_refined = self._masked_attention(
                q_target_refined, filtered_k, filtered_v, filtered_valid
            )                                                                  # Shape: [B, H], [B, k_filter]
            m_target_refined = self.out_target_refined(m_target_refined_raw)   # Shape: [B, H]

        if self.use_gated_fusion:
            features, g_intent, g_refined = self.fusion(m_ally, m_target_refined)  # Shape: [B, H], [B, H], [B, H]
            self.last_g_intent = g_intent
            self.last_g_refined = g_refined
        else:
            features = self.fusion_mlp(torch.cat([m_ally, m_target_refined], dim=-1))  # Shape: [B, H]
            self.last_g_intent = None
            self.last_g_refined = None

        max_pool = max(attn_ally.shape[1], attn_target_refined.shape[1])

        def _pad(t, target):
            if t.shape[1] < target:
                return F.pad(t, (0, target - t.shape[1]))                     # Shape: [B, target]
            return t

        latent_beliefs = torch.stack(
            [_pad(attn_ally, max_pool), _pad(attn_target_refined, max_pool)], dim=1
        )                                                                      # Shape: [B, 2, max_pool]

        if self.use_gated_fusion:
            return features, latent_beliefs, (g_intent, g_refined)
        return features, latent_beliefs

    def get_last_gates(self):
        return self.last_g_intent, self.last_g_refined


class HeteroGraphActorBase(nn.Module):
    """
    Ally -> Enemy/Context -> Ally graph feature extractor.

    The constructor keeps the old landmark_dim/neighbor_dim arguments so
    existing scripts remain usable. Internally, neighbor_dim is used as the
    default ally/enemy feature dimension and landmark_dim as the context
    feature dimension.
    """

    def __init__(
        self,
        agent_state_dim,
        landmark_dim,
        neighbor_dim,
        hidden_size=64,
        k_max=10,
        top_k_filter=5,
        use_knn=True,
        use_am_filter=True,
        use_orthogonal=True,
        use_ReLU=True,
        use_gated_fusion=False,
        ally_dim=None,
        enemy_dim=None,
        ctx_dim=None,
    ):
        super(HeteroGraphActorBase, self).__init__()
        self.agent_state_dim = agent_state_dim
        self.landmark_dim = landmark_dim
        self.neighbor_dim = neighbor_dim
        self.ally_dim = ally_dim or neighbor_dim
        self.enemy_dim = enemy_dim or neighbor_dim
        self.ctx_dim = ctx_dim or landmark_dim
        self.hidden_size = hidden_size
        self.k_max = k_max
        self.top_k_filter = top_k_filter
        self.use_knn = use_knn
        self.use_am_filter = use_am_filter
        self.use_gated_fusion = use_gated_fusion
        self.last_g_intent = None
        self.last_g_refined = None

        H = hidden_size
        act_fn = nn.ReLU() if use_ReLU else nn.Tanh()

        self.ally_perception = GraphAttentionLayer(
            query_dim=agent_state_dim,
            key_dim=self.ally_dim,
            hidden_size=H,
            num_heads=1,
            use_orthogonal=use_orthogonal,
            use_ReLU=use_ReLU,
        )

        self.num_heads2 = 1
        self.head_dim2 = H // self.num_heads2
        self.linear_q2 = nn.Linear(agent_state_dim + H, H)
        self.linear_k2_enemy = nn.Linear(self.enemy_dim, H)
        self.linear_v2_enemy = nn.Linear(self.enemy_dim, H)
        self.linear_k2_ctx = nn.Linear(self.ctx_dim, H)
        self.linear_v2_ctx = nn.Linear(self.ctx_dim, H)
        self.linear_out2 = nn.Linear(H, H)

        self.num_heads3 = 1
        self.head_dim3 = H // self.num_heads3
        self.synergy_mlp = nn.Sequential(
            nn.Linear(H + self.ally_dim, H),
            act_fn,
            nn.Linear(H, 1),
        )
        self.linear_q3 = nn.Linear(agent_state_dim + H, H)
        self.linear_k3 = nn.Linear(self.ally_dim, H)
        self.linear_v3 = nn.Linear(self.ally_dim, H)
        self.linear_out3 = nn.Linear(H, H)
        self._init_manual_layers(use_orthogonal, use_ReLU)

        if self.use_gated_fusion:
            self.fusion = GatedConcatenationFusion(
                dim_intent=H,
                dim_refined=H,
                hidden_size=H,
                use_orthogonal=use_orthogonal,
                use_ReLU=use_ReLU,
            )
        else:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(H * 2, H),
                act_fn,
                nn.LayerNorm(H),
            )
            self._init_fusion(use_orthogonal, use_ReLU)

    def _init_fusion(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        for m in self.fusion_mlp.modules():
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _init_manual_layers(self, use_orthogonal, use_ReLU):
        gain = nn.init.calculate_gain('relu' if use_ReLU else 'tanh')
        modules = [
            self.linear_q2, self.linear_k2_enemy, self.linear_v2_enemy,
            self.linear_k2_ctx, self.linear_v2_ctx, self.linear_out2,
            self.linear_q3, self.linear_k3, self.linear_v3, self.linear_out3,
        ]
        modules.extend([m for m in self.synergy_mlp.modules() if isinstance(m, nn.Linear)])
        for m in modules:
            if isinstance(m, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(m.weight, gain=gain)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _mask_to_valid(mask):
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            return ~mask
        return mask > 0

    def _masked_attention_from_projected(self, Q, K, V, valid_mask, head_dim):
        B = Q.shape[0]
        S = K.shape[-2]
        if S == 0:
            msg = Q.new_zeros(B, self.hidden_size)                            # Shape: [B, H]
            attn_mean = Q.new_zeros(B, 0)                                      # Shape: [B, 0]
            return msg, attn_mean

        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(float(head_dim))  # Shape: [B, Nh, 1, S]
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=Q.device, dtype=torch.bool)      # Shape: [B, S]
            logits = logits.masked_fill(~valid_mask[:, None, None, :], -1e9)   # Shape: [B, Nh, 1, S]
            no_valid = ~valid_mask.any(dim=-1, keepdim=True)                  # Shape: [B, 1]
            logits = logits.masked_fill(no_valid[:, None, :, None], 0.0)      # Shape: [B, Nh, 1, S]

        attn = F.softmax(logits, dim=-1)                                      # Shape: [B, Nh, 1, S]
        if valid_mask is not None:
            keep = valid_mask.to(dtype=attn.dtype)                            # Shape: [B, S]
            attn = attn * keep[:, None, None, :]                              # Shape: [B, Nh, 1, S]
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)           # Shape: [B, Nh, 1, 1]
            attn = attn / denom                                               # Shape: [B, Nh, 1, S]

        out = torch.matmul(attn, V)                                           # Shape: [B, Nh, 1, Dh]
        out = out.transpose(1, 2).contiguous().view(B, self.hidden_size)       # Shape: [B, H]
        if valid_mask is not None:
            out = out * valid_mask.any(dim=-1, keepdim=True).to(dtype=out.dtype)  # Shape: [B, H]
        attn_mean = attn.squeeze(2).mean(dim=1)                               # Shape: [B, S]
        return out, attn_mean

    def forward(self, agent_state, ally_obs, enemy_obs, ctx_obs,
                ally_mask=None, enemy_mask=None, ctx_mask=None):
        """
        Args:
            agent_state: [B, d_agent]
            ally_obs:    [B, N_ally, d_ally]
            enemy_obs:   [B, N_enemy, d_enemy]
            ctx_obs:     [B, N_ctx, d_ctx]
            *_mask: bool masks use True as invalid padding; numeric masks use >0 as valid.

        Returns:
            features:       [B, hidden_size]
            latent_beliefs: [B, 2, max_pool]
                            row-0 = Hop2 threat/context attention
                            row-1 = Hop3 selected-ally attention
        """
        B = agent_state.shape[0]
        H = self.hidden_size
        ally_valid = self._mask_to_valid(ally_mask)
        enemy_valid = self._mask_to_valid(enemy_mask)
        ctx_valid = self._mask_to_valid(ctx_mask)

        # Hop 1: Ally Perception, ego attends over all allies.
        m_ally, _attn_ally = self.ally_perception(
            agent_state, ally_obs, valid_mask=ally_valid
        )                                                                      # Shape: [B, H], [B, N_ally]

        # Hop 2: Threat & Context Alignment, ego + ally formation attends over enemy/context nodes.
        q_threat = torch.cat([agent_state, m_ally], dim=-1)                   # Shape: [B, d_agent+H]
        Q_2 = self.linear_q2(q_threat).view(
            B, 1, self.num_heads2, self.head_dim2
        ).transpose(1, 2)                                                      # Shape: [B, heads, 1, head_dim]

        N_enemy = enemy_obs.shape[1]
        N_ctx = ctx_obs.shape[1]
        K_enemy = self.linear_k2_enemy(enemy_obs).view(
            B, N_enemy, self.num_heads2, self.head_dim2
        ).transpose(1, 2)                                                      # Shape: [B, heads, N_enemy, head_dim]
        K_ctx = self.linear_k2_ctx(ctx_obs).view(
            B, N_ctx, self.num_heads2, self.head_dim2
        ).transpose(1, 2)                                                      # Shape: [B, heads, N_ctx, head_dim]
        V_enemy = self.linear_v2_enemy(enemy_obs).view(
            B, N_enemy, self.num_heads2, self.head_dim2
        ).transpose(1, 2)                                                      # Shape: [B, heads, N_enemy, head_dim]
        V_ctx = self.linear_v2_ctx(ctx_obs).view(
            B, N_ctx, self.num_heads2, self.head_dim2
        ).transpose(1, 2)                                                      # Shape: [B, heads, N_ctx, head_dim]
        K_2 = torch.cat([K_enemy, K_ctx], dim=-2)                              # Shape: [B, heads, N_enemy+N_ctx, head_dim]
        V_2 = torch.cat([V_enemy, V_ctx], dim=-2)                              # Shape: [B, heads, N_enemy+N_ctx, head_dim]
        if enemy_valid is None:
            enemy_valid = agent_state.new_ones(B, N_enemy, dtype=torch.bool)   # Shape: [B, N_enemy]
        if ctx_valid is None:
            ctx_valid = agent_state.new_ones(B, N_ctx, dtype=torch.bool)       # Shape: [B, N_ctx]
        mask_2 = torch.cat([enemy_valid, ctx_valid], dim=-1)                  # Shape: [B, N_enemy+N_ctx]
        m_threat_raw, attn_threat = self._masked_attention_from_projected(
            Q_2, K_2, V_2, mask_2, self.head_dim2
        )                                                                      # Shape: [B, H], [B, N_enemy+N_ctx]
        m_threat = self.linear_out2(m_threat_raw)                              # Shape: [B, H]

        # Hop 3: Cooperative Teammate Selection, ego + threat attends over filtered allies.
        N_ally = ally_obs.shape[1]
        if N_ally == 0:
            m_coop = ally_obs.new_zeros(B, H)                                  # Shape: [B, H]
            attn_coop = ally_obs.new_zeros(B, 0)                               # Shape: [B, 0]
        else:
            if ally_valid is None:
                ally_valid = agent_state.new_ones(B, N_ally, dtype=torch.bool) # Shape: [B, N_ally]
            dist_sq = ally_obs[:, :, 0] ** 2 + ally_obs[:, :, 1] ** 2          # Shape: [B, N_ally]
            dist_sq = dist_sq.masked_fill(~ally_valid, float('inf'))           # Shape: [B, N_ally]
            effective_k = min(self.k_max if self.use_knn else N_ally, N_ally)
            _, knn_idx = torch.topk(
                dist_sq, k=effective_k, dim=-1, largest=False
            )                                                                  # Shape: [B, k_max]
            knn_obs = torch.gather(
                ally_obs, dim=1,
                index=knn_idx.unsqueeze(-1).expand(-1, -1, self.ally_dim)
            )                                                                  # Shape: [B, k_max, d_ally]
            knn_valid = torch.gather(ally_valid, dim=1, index=knn_idx)         # Shape: [B, k_max]

            if self.use_am_filter:
                threat_expand = m_threat.unsqueeze(1).expand(-1, effective_k, -1)  # Shape: [B, k_max, H]
                synergy_in = torch.cat([threat_expand, knn_obs], dim=-1)       # Shape: [B, k_max, H+d_ally]
                synergy_score = self.synergy_mlp(synergy_in).squeeze(-1)       # Shape: [B, k_max]
                synergy_score = synergy_score.masked_fill(~knn_valid, -1e9)   # Shape: [B, k_max]
                filter_k = min(self.top_k_filter, effective_k)
                _, filter_idx = torch.topk(
                    synergy_score, k=filter_k, dim=-1, largest=True
                )                                                              # Shape: [B, k_filter]
                filtered_ally = torch.gather(
                    knn_obs, dim=1,
                    index=filter_idx.unsqueeze(-1).expand(-1, -1, self.ally_dim)
                )                                                              # Shape: [B, k_filter, d_ally]
                filtered_valid = torch.gather(knn_valid, dim=1, index=filter_idx)  # Shape: [B, k_filter]
            else:
                filtered_ally = knn_obs                                        # Shape: [B, k_max, d_ally]
                filtered_valid = knn_valid                                     # Shape: [B, k_max]

            q_coop = torch.cat([agent_state, m_threat], dim=-1)                # Shape: [B, d_agent+H]
            Q_3 = self.linear_q3(q_coop).view(
                B, 1, self.num_heads3, self.head_dim3
            ).transpose(1, 2)                                                  # Shape: [B, heads, 1, head_dim]
            K_3 = self.linear_k3(filtered_ally).view(
                B, filtered_ally.shape[1], self.num_heads3, self.head_dim3
            ).transpose(1, 2)                                                  # Shape: [B, heads, k_filter, head_dim]
            V_3 = self.linear_v3(filtered_ally).view(
                B, filtered_ally.shape[1], self.num_heads3, self.head_dim3
            ).transpose(1, 2)                                                  # Shape: [B, heads, k_filter, head_dim]
            m_coop_raw, attn_coop = self._masked_attention_from_projected(
                Q_3, K_3, V_3, filtered_valid, self.head_dim3
            )                                                                  # Shape: [B, H], [B, k_filter]
            m_coop = self.linear_out3(m_coop_raw)                              # Shape: [B, H]

        # Terminal gated fusion: threat intent + cooperative teammate feature.
        if self.use_gated_fusion:
            features, g_intent, g_refined = self.fusion(
                m_threat, m_coop
            )                                                                  # Shape: [B, H], [B, H], [B, H]
            self.last_g_intent = g_intent
            self.last_g_refined = g_refined
        else:
            fused = torch.cat([m_threat, m_coop], dim=-1)                      # Shape: [B, 2H]
            features = self.fusion_mlp(fused)                                  # Shape: [B, H]
            self.last_g_intent = None
            self.last_g_refined = None

        max_pool = max(attn_threat.shape[1], attn_coop.shape[1])

        def _pad(t, target):
            if t.shape[1] < target:
                return F.pad(t, (0, target - t.shape[1]))                     # Shape: [B, target]
            return t

        latent_beliefs = torch.stack(
            [_pad(attn_threat, max_pool), _pad(attn_coop, max_pool)], dim=1
        )                                                                      # Shape: [B, 2, max_pool]

        if self.use_gated_fusion:
            return features, latent_beliefs, (g_intent, g_refined)
        return features, latent_beliefs

    def get_last_gates(self):
        return self.last_g_intent, self.last_g_refined
