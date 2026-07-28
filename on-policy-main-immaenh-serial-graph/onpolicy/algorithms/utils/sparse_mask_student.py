import torch
import torch.nn as nn


class StudentSparseMaskHead(nn.Module):
    """
    Fast online sparse communication mask predictor over spatial Top-K allies.

    Inputs use per-agent flattened batches: threat_repr [B, H] and candidate
    ally features [B, K, D]. No diffusion sampling is performed here.
    """

    def __init__(
        self,
        threat_dim,
        ally_dim,
        hidden_dim=128,
        edge_top_m=1,
        edge_threshold=0.5,
        use_topm_mask=True,
        use_orthogonal=True,
        use_ReLU=True,
        debug_shapes=False,
    ):
        super(StudentSparseMaskHead, self).__init__()
        self.threat_dim = threat_dim
        self.ally_dim = ally_dim
        self.hidden_dim = hidden_dim
        self.edge_top_m = edge_top_m
        self.edge_threshold = edge_threshold
        self.use_topm_mask = use_topm_mask
        self.debug_shapes = debug_shapes
        self._debug_printed = False

        act_fn = nn.ReLU() if use_ReLU else nn.Tanh()
        self.ally_proj = nn.Linear(ally_dim, threat_dim)
        edge_in_dim = threat_dim * 4 + 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1),
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

    def _hard_mask(self, edge_probs, candidate_mask):
        if edge_probs.shape[1] == 0:
            return edge_probs
        valid = candidate_mask.to(dtype=torch.bool)
        if self.use_topm_mask:
            scores = edge_probs.masked_fill(~valid, -1.0)
            k = min(max(int(self.edge_top_m), 1), edge_probs.shape[1])
            _, idx = torch.topk(scores, k=k, dim=-1, largest=True)
            hard = torch.zeros_like(edge_probs)
            hard.scatter_(dim=-1, index=idx, value=1.0)
            hard = hard * valid.to(dtype=hard.dtype)
        else:
            hard = (edge_probs > self.edge_threshold).to(dtype=edge_probs.dtype)
            hard = hard * valid.to(dtype=hard.dtype)
        return hard

    def forward(self, threat_repr, candidate_ally, candidate_mask, distance_feat=None):
        B, K = candidate_ally.shape[:2]
        if K == 0:
            empty = candidate_ally.new_zeros(B, 0)
            return {
                "edge_logits": empty,
                "edge_probs": empty,
                "hard_mask": empty,
                "edge_context": candidate_ally.new_zeros(B, 0, self.threat_dim * 4 + 3),
                "student_debug": {},
            }

        ally_h = self.ally_proj(candidate_ally)                         # [B, K, H]
        threat = threat_repr.unsqueeze(1).expand(-1, K, -1)             # [B, K, H]
        if distance_feat is None:
            rel = candidate_ally[..., :2] if candidate_ally.shape[-1] >= 2 else candidate_ally.new_zeros(B, K, 2)
            dist = torch.norm(rel, dim=-1, keepdim=True)
            distance_feat = torch.cat([dist, rel], dim=-1)              # [B, K, 3]

        edge_context = torch.cat(
            [threat, ally_h, threat * ally_h, torch.abs(threat - ally_h), distance_feat],
            dim=-1,
        )                                                               # [B, K, 4H+3]
        edge_logits = self.edge_mlp(edge_context).squeeze(-1)           # [B, K]
        valid = candidate_mask.to(dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(~valid, -1e9)
        edge_probs = torch.sigmoid(edge_logits) * valid.to(dtype=edge_logits.dtype)
        hard_mask = self._hard_mask(edge_probs, valid)                  # [B, K]

        if self.debug_shapes and not self._debug_printed:
            print("[SMD] threat_repr shape", tuple(threat_repr.shape))
            print("[SMD] ally_repr shape", tuple(candidate_ally.shape))
            print("[SMD] candidate_mask shape", tuple(candidate_mask.shape))
            print("[SMD] edge_context shape", tuple(edge_context.shape))
            print("[SMD] edge_logits shape", tuple(edge_logits.shape))
            print("[SMD] hard_mask shape", tuple(hard_mask.shape))
            self._debug_printed = True

        return {
            "edge_logits": edge_logits,
            "edge_probs": edge_probs,
            "hard_mask": hard_mask,
            "edge_context": edge_context,
            "student_debug": {},
        }
