import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


NEG_INF = -1.0e9


def _as_bool_valid(valid_mask):
    if valid_mask is None:
        return None
    if valid_mask.dtype == torch.bool:
        return valid_mask
    return valid_mask > 0


def budget_topm_mask(scores, valid_mask, edge_m):
    valid = _as_bool_valid(valid_mask)
    if valid is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    if scores.shape[-1] == 0 or int(edge_m) <= 0:
        return scores.new_zeros(scores.shape)
    k = min(int(edge_m), scores.shape[-1])
    masked_scores = scores.masked_fill(~valid, NEG_INF)
    _, idx = torch.topk(masked_scores, k=k, dim=-1, largest=True)
    hard = torch.zeros_like(scores)
    hard.scatter_(dim=-1, index=idx, value=1.0)
    hard = hard * valid.to(dtype=hard.dtype)
    return hard


def select_relation_candidates(entity_features, relative_positions, valid_mask, top_k):
    B = entity_features.shape[0]
    K = int(top_k)
    valid = _as_bool_valid(valid_mask)
    if valid is None:
        valid = entity_features.new_ones(entity_features.shape[:2], dtype=torch.bool)
    if K <= 0 or entity_features.shape[1] == 0:
        return {
            "candidate_features": entity_features.new_zeros(B, 0, entity_features.shape[-1]),
            "candidate_relative_positions": relative_positions.new_zeros(B, 0, relative_positions.shape[-1]),
            "candidate_distances": relative_positions.new_zeros(B, 0),
            "candidate_valid_mask": valid.new_zeros(B, 0),
            "candidate_original_indices": torch.zeros(B, 0, dtype=torch.long, device=entity_features.device),
        }
    distances = torch.norm(relative_positions[..., :2], dim=-1)
    distances = distances.masked_fill(~valid, float("inf"))
    k = min(K, entity_features.shape[1])
    cand_dist, idx = torch.topk(distances, k=k, dim=-1, largest=False)
    feature_idx = idx.unsqueeze(-1).expand(-1, -1, entity_features.shape[-1])
    rel_idx = idx.unsqueeze(-1).expand(-1, -1, relative_positions.shape[-1])
    cand_features = torch.gather(entity_features, dim=1, index=feature_idx)
    cand_rel = torch.gather(relative_positions, dim=1, index=rel_idx)
    cand_valid = torch.gather(valid, dim=1, index=idx)
    cand_features = cand_features * cand_valid.unsqueeze(-1).to(dtype=cand_features.dtype)
    cand_rel = cand_rel * cand_valid.unsqueeze(-1).to(dtype=cand_rel.dtype)
    cand_dist = cand_dist.masked_fill(~cand_valid, 0.0)
    return {
        "candidate_features": cand_features,
        "candidate_relative_positions": cand_rel,
        "candidate_distances": cand_dist,
        "candidate_valid_mask": cand_valid,
        "candidate_original_indices": idx,
    }


def build_budget_operation_distribution(drop_logits, add_logits, noop_logits, current_mask, valid_mask, pair_logits=None):
    valid = _as_bool_valid(valid_mask)
    if valid is None:
        valid = torch.ones_like(current_mask, dtype=torch.bool)
    B, K = current_mask.shape
    selected = (current_mask > 0.5) & valid
    unselected = (~selected) & valid
    max_pairs = int((selected.sum(-1) * unselected.sum(-1)).max().item()) if K > 0 else 0
    op_count = max(max_pairs + 1, 1)
    operation_logits = drop_logits.new_full((B, op_count), NEG_INF)
    drop_indices = torch.full((B, op_count), -1, dtype=torch.long, device=current_mask.device)
    add_indices = torch.full((B, op_count), -1, dtype=torch.long, device=current_mask.device)
    valid_operations = torch.zeros(B, op_count, dtype=torch.bool, device=current_mask.device)
    if noop_logits.dim() > 1:
        noop_values = noop_logits.squeeze(-1)
    else:
        noop_values = noop_logits

    all_indices = torch.arange(K, device=current_mask.device)
    for b in range(B):
        drops = all_indices[selected[b]]
        adds = all_indices[unselected[b]]
        cursor = 0
        for d in drops:
            for a in adds:
                pair = drop_logits[b, d] + add_logits[b, a]
                if pair_logits is not None:
                    pair = pair + pair_logits[b, d, a]
                operation_logits[b, cursor] = pair
                drop_indices[b, cursor] = d
                add_indices[b, cursor] = a
                valid_operations[b, cursor] = True
                cursor += 1
        operation_logits[b, cursor] = noop_values[b]
        valid_operations[b, cursor] = True
    return {
        "operation_logits": operation_logits,
        "drop_indices": drop_indices,
        "add_indices": add_indices,
        "valid_operations": valid_operations,
    }


def _one_hot_from_indices(indices, K, dtype):
    out = torch.zeros(indices.shape[0], K, dtype=dtype, device=indices.device)
    valid = indices >= 0
    if valid.any():
        out.scatter_(1, indices.clamp_min(0).unsqueeze(-1), valid.to(dtype=dtype).unsqueeze(-1))
    return out


def apply_budget_operation(current_mask, valid_mask, operation, hard_operation_index=None):
    valid = _as_bool_valid(valid_mask)
    if valid is None:
        valid = torch.ones_like(current_mask, dtype=torch.bool)
    logits = operation["operation_logits"].masked_fill(~operation["valid_operations"], NEG_INF)
    probs = F.softmax(logits, dim=-1) * operation["valid_operations"].to(dtype=logits.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    drop_indices = operation["drop_indices"]
    add_indices = operation["add_indices"]
    B, K = current_mask.shape
    if K == 0:
        return current_mask, current_mask
    if hard_operation_index is None:
        hard_operation_index = logits.argmax(dim=-1)
    chosen_drop = torch.gather(drop_indices, 1, hard_operation_index.view(-1, 1)).squeeze(-1)
    chosen_add = torch.gather(add_indices, 1, hard_operation_index.view(-1, 1)).squeeze(-1)
    hard = current_mask - _one_hot_from_indices(chosen_drop, K, current_mask.dtype)
    hard = hard + _one_hot_from_indices(chosen_add, K, current_mask.dtype)
    hard = hard.clamp(0.0, 1.0) * valid.to(dtype=hard.dtype)

    valid_pair = (drop_indices >= 0) & (add_indices >= 0) & operation["valid_operations"]
    pair_probs = probs * valid_pair.to(dtype=probs.dtype)
    p_drop = current_mask.new_zeros(B, K)
    p_add = current_mask.new_zeros(B, K)
    if drop_indices.numel() > 0:
        p_drop.scatter_add_(1, drop_indices.clamp_min(0), pair_probs)
        p_add.scatter_add_(1, add_indices.clamp_min(0), pair_probs)
    soft = (current_mask - p_drop + p_add) * valid.to(dtype=current_mask.dtype)
    return hard, soft


def straight_through_budget_mask(hard_mask, soft_mask):
    return hard_mask.detach() - soft_mask.detach() + soft_mask


def operation_target_index(operation, drop_index, add_index, noop):
    B = operation["operation_logits"].shape[0]
    device = operation["operation_logits"].device
    if isinstance(noop, bool):
        noop_tensor = torch.full((B,), noop, dtype=torch.bool, device=device)
    else:
        noop_tensor = noop.to(device=device, dtype=torch.bool).view(-1)
    if drop_index is None:
        drop_index = torch.full((B,), -1, dtype=torch.long, device=device)
    if add_index is None:
        add_index = torch.full((B,), -1, dtype=torch.long, device=device)
    drop_index = drop_index.to(device=device, dtype=torch.long).view(-1)
    add_index = add_index.to(device=device, dtype=torch.long).view(-1)
    out = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        if noop_tensor[b]:
            match = (operation["drop_indices"][b] < 0) & (operation["add_indices"][b] < 0) & operation["valid_operations"][b]
        else:
            match = (
                (operation["drop_indices"][b] == drop_index[b])
                & (operation["add_indices"][b] == add_index[b])
                & operation["valid_operations"][b]
            )
        idx = torch.nonzero(match, as_tuple=False)
        if idx.numel() == 0:
            raise ValueError("target operation is not legal for current mask")
        out[b] = idx[0, 0]
    return out


def corrupt_target_mask(target_mask, valid_mask, corruption_prob=1.0, generator=None):
    valid = _as_bool_valid(valid_mask)
    B, K = target_mask.shape
    corrupted = target_mask.clone()
    reverse_drop = torch.full((B,), -1, dtype=torch.long, device=target_mask.device)
    reverse_add = torch.full((B,), -1, dtype=torch.long, device=target_mask.device)
    did_corrupt = torch.zeros(B, dtype=torch.bool, device=target_mask.device)
    all_indices = torch.arange(K, device=target_mask.device)
    rand = torch.rand(B, device=target_mask.device, generator=generator)
    for b in range(B):
        selected = all_indices[(target_mask[b] > 0.5) & valid[b]]
        unselected = all_indices[(target_mask[b] <= 0.5) & valid[b]]
        if rand[b] >= corruption_prob or selected.numel() == 0 or unselected.numel() == 0:
            continue
        d = selected[torch.randint(selected.numel(), (1,), device=target_mask.device, generator=generator)[0]]
        a = unselected[torch.randint(unselected.numel(), (1,), device=target_mask.device, generator=generator)[0]]
        corrupted[b, d] = 0.0
        corrupted[b, a] = 1.0
        reverse_drop[b] = a
        reverse_add[b] = d
        did_corrupt[b] = True
    return {
        "corrupted_mask": corrupted,
        "reverse_drop_index": reverse_drop,
        "reverse_add_index": reverse_add,
        "did_corrupt": did_corrupt,
    }


def ppo_surrogate(new_log_probs, old_log_probs, advantages, clip_param):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    return torch.min(surr1, surr2)


def select_best_target_mask(candidate_masks, candidate_surrogates, base_index=0, margin=0.0):
    values = candidate_surrogates
    while values.dim() > 2:
        values = values.squeeze(-1)
    target_idx = values.argmax(dim=1)
    base_value = values[:, base_index]
    target_value = values.gather(1, target_idx.unsqueeze(-1)).squeeze(-1)
    changed = target_value > (base_value + margin)
    final_idx = torch.where(changed, target_idx, torch.full_like(target_idx, base_index))
    gather_idx = final_idx.view(-1, 1, 1).expand(-1, 1, candidate_masks.shape[-1])
    chosen = torch.gather(candidate_masks, 1, gather_idx).squeeze(1)
    return chosen, {
        "target_index": final_idx,
        "target_changed": changed,
        "base_surrogate": base_value.detach(),
        "target_surrogate": target_value.detach(),
        "surrogate_improvement": (target_value - base_value).detach(),
    }


class RelationEncoder(nn.Module):
    def __init__(self, self_dim, hop_dim, entity_dim, out_dim, use_ReLU=True):
        super(RelationEncoder, self).__init__()
        act = nn.ReLU() if use_ReLU else nn.Tanh()
        in_dim = self_dim + hop_dim * 2 + entity_dim + 3
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), act, nn.Linear(out_dim, out_dim))

    def forward(self, self_feature, hop1_feature, hop2_feature, entity_features, relative_positions, valid_mask):
        B, K = entity_features.shape[:2]
        if K == 0:
            return entity_features.new_zeros(B, 0, self.net[-1].out_features)
        rel = relative_positions[..., :2]
        dist = torch.norm(rel, dim=-1, keepdim=True)
        self_expand = self_feature.unsqueeze(1).expand(-1, K, -1)
        hop1_expand = hop1_feature.unsqueeze(1).expand(-1, K, -1)
        hop2_expand = hop2_feature.unsqueeze(1).expand(-1, K, -1)
        token_in = torch.cat([self_expand, hop1_expand, hop2_expand, entity_features, dist, rel], dim=-1)
        valid = _as_bool_valid(valid_mask)
        tokens = self.net(token_in)
        if valid is not None:
            tokens = tokens * valid.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens


class JointBudgetPreservingSubgraphDenoiser(nn.Module):
    def __init__(
        self,
        relation_dim,
        condition_dim,
        hidden_dim=64,
        num_heads=2,
        num_layers=1,
        type_embedding_dim=8,
        state_embedding_dim=8,
        time_embedding_dim=16,
    ):
        super(JointBudgetPreservingSubgraphDenoiser, self).__init__()
        self.hidden_dim = hidden_dim
        self.type_embedding = nn.Embedding(2, type_embedding_dim)
        self.state_embedding = nn.Embedding(2, state_embedding_dim)
        self.time_embedding = nn.Embedding(8, time_embedding_dim)
        self.relation_proj = nn.Linear(relation_dim, hidden_dim)
        self.meta_proj = nn.Linear(type_embedding_dim + state_embedding_dim + time_embedding_dim + condition_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ally_drop_head = nn.Linear(hidden_dim, 1)
        self.ally_add_head = nn.Linear(hidden_dim, 1)
        self.enemy_drop_head = nn.Linear(hidden_dim, 1)
        self.enemy_add_head = nn.Linear(hidden_dim, 1)
        self.ally_noop_head = nn.Linear(condition_dim, 1)
        self.enemy_noop_head = nn.Linear(condition_dim, 1)

    def _pack(self, tokens, current_mask, valid_mask, type_id, condition, timestep):
        B, K = tokens.shape[:2]
        if K == 0:
            return tokens.new_zeros(B, 0, self.hidden_dim)
        type_ids = torch.full((B, K), type_id, dtype=torch.long, device=tokens.device)
        state_ids = (current_mask > 0.5).long()
        if not torch.is_tensor(timestep):
            timestep = torch.full((B,), int(timestep), dtype=torch.long, device=tokens.device)
        t_emb = self.time_embedding(timestep.long().clamp(max=self.time_embedding.num_embeddings - 1)).unsqueeze(1).expand(-1, K, -1)
        meta = torch.cat(
            [
                self.type_embedding(type_ids),
                self.state_embedding(state_ids),
                t_emb,
                condition.unsqueeze(1).expand(-1, K, -1),
            ],
            dim=-1,
        )
        return self.relation_proj(tokens) + self.meta_proj(meta)

    def forward(
        self,
        ally_relation_tokens,
        enemy_relation_tokens,
        ally_current_mask,
        enemy_current_mask,
        ally_valid_mask,
        enemy_valid_mask,
        condition,
        diffusion_timestep=0,
    ):
        ally_h = self._pack(ally_relation_tokens, ally_current_mask, ally_valid_mask, 0, condition, diffusion_timestep)
        enemy_h = self._pack(enemy_relation_tokens, enemy_current_mask, enemy_valid_mask, 1, condition, diffusion_timestep)
        joint = torch.cat([ally_h, enemy_h], dim=1)
        ally_valid = _as_bool_valid(ally_valid_mask)
        enemy_valid = _as_bool_valid(enemy_valid_mask)
        padding = ~torch.cat([ally_valid, enemy_valid], dim=1) if joint.shape[1] > 0 else None
        if joint.shape[1] > 0:
            all_pad = padding.all(dim=1) if padding is not None else torch.zeros(joint.shape[0], dtype=torch.bool, device=joint.device)
            padding = padding.masked_fill(all_pad.unsqueeze(-1), False)
            joint = self.transformer(joint, src_key_padding_mask=padding)
        K_a = ally_relation_tokens.shape[1]
        ally_out = joint[:, :K_a, :] if K_a > 0 else ally_h
        enemy_out = joint[:, K_a:, :] if enemy_relation_tokens.shape[1] > 0 else enemy_h
        B = ally_relation_tokens.shape[0]
        return {
            "ally_drop_logits": self.ally_drop_head(ally_out).squeeze(-1) if K_a > 0 else ally_relation_tokens.new_zeros(B, 0),
            "ally_add_logits": self.ally_add_head(ally_out).squeeze(-1) if K_a > 0 else ally_relation_tokens.new_zeros(B, 0),
            "ally_noop_logits": self.ally_noop_head(condition).squeeze(-1),
            "enemy_drop_logits": self.enemy_drop_head(enemy_out).squeeze(-1) if enemy_relation_tokens.shape[1] > 0 else enemy_relation_tokens.new_zeros(B, 0),
            "enemy_add_logits": self.enemy_add_head(enemy_out).squeeze(-1) if enemy_relation_tokens.shape[1] > 0 else enemy_relation_tokens.new_zeros(B, 0),
            "enemy_noop_logits": self.enemy_noop_head(condition).squeeze(-1),
        }


class MaskedRelationAttention(nn.Module):
    def __init__(self, query_dim, token_dim, value_dim, hidden_dim, num_heads=1):
        super(MaskedRelationAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(token_dim, hidden_dim)
        self.v_proj = nn.Linear(value_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, relation_tokens, values, mask, valid_mask):
        B, K = values.shape[:2]
        if K == 0:
            return query.new_zeros(B, self.hidden_dim), query.new_zeros(B, 0)
        valid = _as_bool_valid(valid_mask)
        if valid is None:
            valid = torch.ones(B, K, dtype=torch.bool, device=query.device)
        keep = (mask > 0) & valid
        mask_weight = mask.clamp_min(0.0) * valid.to(dtype=mask.dtype)
        q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(relation_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        logits = logits.masked_fill(~keep[:, None, None, :], NEG_INF)
        no_keep = ~keep.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(no_keep[:, None, :, None], 0.0)
        attn = F.softmax(logits, dim=-1)
        attn = attn * mask_weight[:, None, None, :].to(dtype=attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, self.hidden_dim)
        out = self.out_proj(out)
        out = out * keep.any(dim=-1, keepdim=True).to(dtype=out.dtype)
        return out, attn.squeeze(2).mean(dim=1)


class AllyEnemyInteractionFusion(nn.Module):
    def __init__(self, self_dim, hidden_dim, use_ReLU=True):
        super(AllyEnemyInteractionFusion, self).__init__()
        act = nn.ReLU() if use_ReLU else nn.Tanh()
        self.self_projection = nn.Linear(self_dim, hidden_dim)
        self.ally_projection = nn.Linear(hidden_dim, hidden_dim)
        self.enemy_projection = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_mlp = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), act, nn.Linear(hidden_dim, hidden_dim))
        self.interaction_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(self_dim + hidden_dim * 2, 3)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, self_feature, ally_context, enemy_context):
        interaction_in = torch.cat(
            [
                ally_context,
                enemy_context,
                ally_context * enemy_context,
                torch.abs(ally_context - enemy_context),
            ],
            dim=-1,
        )
        interaction = self.interaction_mlp(interaction_in)
        gates = F.softmax(self.gate(torch.cat([self_feature, ally_context, enemy_context], dim=-1)), dim=-1)
        out = (
            self.self_projection(self_feature)
            + gates[:, 0:1] * self.ally_projection(ally_context)
            + gates[:, 1:2] * self.enemy_projection(enemy_context)
            + gates[:, 2:3] * self.interaction_projection(interaction)
        )
        return self.norm(out), gates
