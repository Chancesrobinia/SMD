"""
Ego-centric Sparse Multiplex Graph (ESMG) module.

Implements:
  - KNN physical truncation (avoid O(N^2))       — toggled by use_knn
  - AM Filter (value-guided adaptive pruning)     — toggled by use_am_filter
  - Multi-layer parallel graph attention (latent beliefs)
  - Message passing aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EgoSparseMultiplexGraph(nn.Module):
    """
    ESMG: Value-Guided Sparse Multiplex Graph for decentralized multi-agent RL.

    Args:
        self_dim (int): dimension of ego-agent observation.
        neighbor_dim (int): dimension of each neighbor's observation (first 2 dims are dx, dy).
        hidden_size (int): hidden feature dimension.
        k_max (int): max neighbors after KNN physical truncation.
        top_k_filter (int): neighbors kept after AM value filter.
        num_layers (int): number of parallel graph attention layers (multiplex layers).
        use_orthogonal (bool): whether to use orthogonal init.
        use_ReLU (bool): whether to use ReLU (else Tanh).
        use_knn (bool): whether to use KNN physical truncation. If False, all neighbors are kept.
        use_am_filter (bool): whether to use AM value filter + hard pruning. If False, all
                              (post-KNN) neighbors pass through with uniform mask (0.0).
    """

    def __init__(
        self,
        self_dim,
        neighbor_dim,
        hidden_size=64,
        k_max=10,
        top_k_filter=5,
        num_layers=2,
        use_orthogonal=True,
        use_ReLU=True,
        use_knn=True,
        use_am_filter=True,
    ):
        super(EgoSparseMultiplexGraph, self).__init__()
        self.self_dim = self_dim
        self.neighbor_dim = neighbor_dim
        self.hidden_size = hidden_size
        self.k_max = k_max
        self.top_k_filter = top_k_filter
        self.num_layers = num_layers
        self.use_knn = use_knn
        self.use_am_filter = use_am_filter

        active_func = nn.ReLU() if use_ReLU else nn.Tanh()

        # ---- Self encoder ----
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            active_func,
            nn.LayerNorm(hidden_size),
        )

        # ---- Neighbor encoder ----
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_size),
            active_func,
            nn.LayerNorm(hidden_size),
        )

        # ---- AM Filter: MLP -> Sigmoid producing scalar value weight per neighbor ----
        # Always instantiated so state_dict keys are stable; forward is gated by use_am_filter.
        self.am_filter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            active_func,
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # ---- Multiplex graph attention: num_layers parallel edge-logit projections ----
        self.edge_logit_layers = nn.ModuleList([
            nn.Linear(hidden_size * 2, 1) for _ in range(num_layers)
        ])

        # ---- Output projection: concat of num_layers aggregated messages + self_h ----
        out_input_dim = hidden_size * num_layers + hidden_size
        self.output_layer = nn.Sequential(
            nn.Linear(out_input_dim, hidden_size),
            active_func,
            nn.LayerNorm(hidden_size),
        )

        # Apply weight init
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if use_orthogonal:
                    nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu' if use_ReLU else 'tanh'))
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, self_obs, neighbor_obs):
        """
        Args:
            self_obs:     [batch, self_dim]
            neighbor_obs: [batch, N, neighbor_dim]  (first 2 cols are dx, dy)

        Returns:
            out_features:   [batch, hidden_size]
            latent_beliefs: [batch, num_layers, effective_pool]  (attention weights per layer)
                            effective_pool is top_k_filter (clamped) when AM filter on,
                            else effective_k_max (all post-KNN neighbors).
        """
        batch_size = self_obs.shape[0]
        N = neighbor_obs.shape[1]  # total number of neighbors in observation

        # ==================================================================
        # 1. KNN Physical Truncation (or bypass)
        # ==================================================================
        if self.use_knn:
            effective_k_max = min(self.k_max, N)
            dx = neighbor_obs[:, :, 0]  # [batch, N]
            dy = neighbor_obs[:, :, 1]  # [batch, N]
            dist_sq = dx ** 2 + dy ** 2  # [batch, N]

            _, knn_indices = torch.topk(dist_sq, k=effective_k_max, dim=-1, largest=False)
            knn_indices_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, self.neighbor_dim)
            knn_obs = torch.gather(neighbor_obs, dim=1, index=knn_indices_expanded)
        else:
            # No KNN truncation — use all neighbors
            effective_k_max = N
            knn_obs = neighbor_obs  # [batch, N, neighbor_dim]

        # ==================================================================
        # 2. Encode features
        # ==================================================================
        self_h = self.self_encoder(self_obs)          # [batch, hidden_size]
        neighbor_h = self.neighbor_encoder(knn_obs)   # [batch, effective_k_max, hidden_size]

        # ==================================================================
        # 3. AM Filter — value-guided pruning (or bypass)
        # ==================================================================
        if self.use_am_filter:
            effective_top_k = min(self.top_k_filter, effective_k_max)

            w = self.am_filter(neighbor_h).squeeze(-1)  # [batch, effective_k_max]
            _, filter_indices = torch.topk(w, k=effective_top_k, dim=-1, largest=True)

            # Hard pruning mask: 0.0 for selected, -inf for pruned
            mask = torch.full(
                (batch_size, effective_k_max), float('-inf'),
                device=self_obs.device, dtype=self_obs.dtype
            )
            mask.scatter_(1, filter_indices, 0.0)
            effective_pool = effective_top_k
        else:
            # No pruning — mask is all 0.0, all neighbors pass through
            mask = torch.zeros(
                (batch_size, effective_k_max),
                device=self_obs.device, dtype=self_obs.dtype
            )
            # filter_indices covers all positions (for latent_beliefs gathering)
            filter_indices = torch.arange(
                effective_k_max, device=self_obs.device
            ).unsqueeze(0).expand(batch_size, -1)
            effective_pool = effective_k_max

        # ==================================================================
        # 4. Multiplex Graph Attention (K parallel layers)
        # ==================================================================
        self_h_expanded = self_h.unsqueeze(1).expand(-1, effective_k_max, -1)
        pair_features = torch.cat([self_h_expanded, neighbor_h], dim=-1)  # [batch, effective_k_max, hidden*2]

        latent_beliefs_list = []
        aggregated_msgs = []

        for layer_idx in range(self.num_layers):
            edge_logits = self.edge_logit_layers[layer_idx](pair_features).squeeze(-1)  # [batch, effective_k_max]
            masked_logits = edge_logits + mask
            z_k = F.softmax(masked_logits, dim=-1)  # [batch, effective_k_max]

            # Extract beliefs at the selected positions
            z_k_filtered = torch.gather(z_k, dim=1, index=filter_indices)  # [batch, effective_pool]
            latent_beliefs_list.append(z_k_filtered)

            # Message passing
            z_k_expanded = z_k.unsqueeze(-1)  # [batch, effective_k_max, 1]
            msg = (z_k_expanded * neighbor_h).sum(dim=1)  # [batch, hidden_size]
            aggregated_msgs.append(msg)

        # Stack latent beliefs: [batch, num_layers, effective_pool]
        latent_beliefs = torch.stack(latent_beliefs_list, dim=1)

        # ==================================================================
        # 5. Output aggregation
        # ==================================================================
        all_msgs = torch.cat(aggregated_msgs, dim=-1)        # [batch, hidden_size * num_layers]
        combined = torch.cat([self_h, all_msgs], dim=-1)      # [batch, hidden_size * (num_layers + 1)]
        out_features = self.output_layer(combined)             # [batch, hidden_size]

        return out_features, latent_beliefs
