import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.utils.gsd_bsd import operation_target_index

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # ESMG reconstruction loss
        self._use_esmg = getattr(args, 'self_dim', None) is not None and getattr(args, 'neighbor_dim', None) is not None
        self._use_am_filter = getattr(args, 'use_am_filter', True)
        self.recon_loss_coef = getattr(args, 'recon_loss_coef', 0.01)
        self._use_smd = getattr(args, 'use_smd', False)
        self._use_gsd_bsd = getattr(args, 'use_gsd_bsd', False)
        self.lambda_gsd_bsd_diff = getattr(args, 'lambda_gsd_bsd_diff', 0.01)
        self.lambda_gsd_bsd_noop = getattr(args, 'lambda_gsd_bsd_noop', 1.0)
        self.lambda_gsd_bsd_action_consistency = getattr(args, 'lambda_gsd_bsd_action_consistency', 0.0)

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def _build_smd_pseudo_mask(self, edge_probs, candidate_mask, adv_targ, pseudo_top_m, positive_only=False):
        valid = candidate_mask.to(dtype=torch.bool)
        if edge_probs.shape[1] == 0:
            return edge_probs.new_zeros(edge_probs.shape)
        adv = adv_targ.detach()
        while adv.dim() > 1:
            adv = adv.squeeze(-1)
        adv_weight = torch.abs(adv).view(-1, 1).to(dtype=edge_probs.dtype)
        utility = edge_probs.detach() * adv_weight
        if positive_only:
            utility = utility * (adv.view(-1, 1) > 0).to(dtype=edge_probs.dtype)
        utility = utility.masked_fill(~valid, -1.0)
        k = min(max(int(pseudo_top_m), 1), edge_probs.shape[1])
        _, idx = torch.topk(utility, k=k, dim=-1, largest=True)
        pseudo = torch.zeros_like(edge_probs)
        pseudo.scatter_(dim=-1, index=idx, value=1.0)
        pseudo = pseudo * valid.to(dtype=pseudo.dtype)
        if positive_only:
            pseudo = pseudo * (adv.view(-1, 1) > 0).to(dtype=pseudo.dtype)
        return pseudo.detach()

    def compute_smd_aux_loss(self, smd_aux, adv_targ):
        if not smd_aux:
            z = torch.tensor(0.0, device=self.device)
            return z, {
                'smd_loss': 0.0,
                'mask_diffusion_loss': 0.0,
                'mask_distill_loss': 0.0,
                'sparse_loss': 0.0,
                'smd_valid_ratio': 0.0,
                'smd_avg_degree': 0.0,
                'smd_edge_prob_mean': 0.0,
                'smd_edge_prob_max': 0.0,
                'smd_hard_edge_ratio': 0.0,
                'smd_pseudo_positive_ratio': 0.0,
            }
        edge_logits = smd_aux['edge_logits']
        edge_probs = smd_aux['edge_probs']
        hard_mask = smd_aux['hard_mask']
        candidate_mask = smd_aux['candidate_mask'].to(device=edge_logits.device, dtype=edge_logits.dtype)
        edge_context = smd_aux['edge_context']
        base = getattr(self.policy.actor, 'base', None)

        pseudo_mask = self._build_smd_pseudo_mask(
            edge_probs, candidate_mask, adv_targ,
            getattr(base, 'smd_pseudo_top_m', 1),
            getattr(base, 'smd_use_positive_adv_only', False),
        )
        valid = candidate_mask.to(dtype=torch.bool)
        bce = nn.functional.binary_cross_entropy_with_logits(edge_logits, pseudo_mask, reduction='none')
        mask_distill_loss = (bce * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)

        if getattr(base, 'smd_teacher', None) is not None:
            teacher_out = base.smd_teacher(edge_context, pseudo_mask, candidate_mask)
            mask_diffusion_loss = teacher_out['mask_diffusion_loss']
        else:
            mask_diffusion_loss = edge_logits.new_tensor(0.0)

        degree = (hard_mask * candidate_mask).sum(dim=-1)
        valid_agent = valid.any(dim=-1).to(dtype=edge_logits.dtype)
        avg_degree = (degree * valid_agent).sum() / valid_agent.sum().clamp_min(1.0)
        sparse_loss = torch.relu(avg_degree - getattr(base, 'smd_target_degree', 1.0))

        smd_loss = (
            getattr(base, 'lambda_smd_diff', 0.01) * mask_diffusion_loss
            + getattr(base, 'lambda_smd_distill', 0.05) * mask_distill_loss
            + getattr(base, 'lambda_smd_sparse', 0.001) * sparse_loss
        )
        valid_ratio = candidate_mask.mean().detach() if candidate_mask.numel() > 0 else edge_logits.new_tensor(0.0)
        edge_prob_valid = edge_probs[valid] if valid.any() else edge_probs.new_zeros(1)
        hard_ratio = ((hard_mask * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)).detach()
        pseudo_ratio = (pseudo_mask.sum() / candidate_mask.sum().clamp_min(1.0)).detach()
        info = {
            'smd_loss': smd_loss.detach().item(),
            'mask_diffusion_loss': mask_diffusion_loss.detach().item(),
            'mask_distill_loss': mask_distill_loss.detach().item(),
            'sparse_loss': sparse_loss.detach().item(),
            'smd_valid_ratio': valid_ratio.item(),
            'smd_avg_degree': avg_degree.detach().item(),
            'smd_edge_prob_mean': edge_prob_valid.mean().detach().item(),
            'smd_edge_prob_max': edge_prob_valid.max().detach().item(),
            'smd_hard_edge_ratio': hard_ratio.item(),
            'smd_pseudo_positive_ratio': pseudo_ratio.item(),
        }
        return smd_loss, info

    def compute_gsd_bsd_aux_loss(self, bsd_aux):
        if not bsd_aux:
            zero = torch.tensor(0.0, device=self.device)
            info = {
                'gsd_bsd/ally_diffusion_loss': 0.0,
                'gsd_bsd/enemy_diffusion_loss': 0.0,
                'gsd_bsd/total_diffusion_loss': 0.0,
                'gsd_bsd/ally_valid_count': 0.0,
                'gsd_bsd/enemy_valid_count': 0.0,
                'gsd_bsd/ally_selected_count': 0.0,
                'gsd_bsd/enemy_selected_count': 0.0,
                'gsd_bsd/ally_swap_rate': 0.0,
                'gsd_bsd/enemy_swap_rate': 0.0,
                'gsd_bsd/ally_noop_rate': 0.0,
                'gsd_bsd/enemy_noop_rate': 0.0,
                'gsd_bsd/ally_operation_entropy': 0.0,
                'gsd_bsd/enemy_operation_entropy': 0.0,
                'gsd_bsd/ally_soft_budget_error': 0.0,
                'gsd_bsd/enemy_soft_budget_error': 0.0,
                'gsd_bsd/ally_mask_change_rate': 0.0,
                'gsd_bsd/enemy_mask_change_rate': 0.0,
                'gsd_bsd/base_surrogate': 0.0,
                'gsd_bsd/target_surrogate': 0.0,
                'gsd_bsd/surrogate_improvement': 0.0,
                'gsd_bsd/target_changed_rate': 0.0,
                'gsd_bsd/target_search_candidates': 0.0,
                'gsd_bsd/target_search_time_ms': 0.0,
                'gsd_bsd/denoiser_time_ms': 0.0,
            }
            return zero, info

        def _branch_loss(prefix):
            operation = bsd_aux.get(f'{prefix}_operation', {})
            operation_logits = bsd_aux.get(f'{prefix}_operation_logits')
            if operation_logits is None or operation_logits.numel() == 0:
                zero = torch.tensor(0.0, device=self.device)
                return zero, {
                    f'gsd_bsd/{prefix}_diffusion_loss': 0.0,
                    f'gsd_bsd/{prefix}_valid_count': 0.0,
                    f'gsd_bsd/{prefix}_selected_count': 0.0,
                    f'gsd_bsd/{prefix}_swap_rate': 0.0,
                    f'gsd_bsd/{prefix}_noop_rate': 0.0,
                    f'gsd_bsd/{prefix}_operation_entropy': 0.0,
                    f'gsd_bsd/{prefix}_soft_budget_error': 0.0,
                    f'gsd_bsd/{prefix}_mask_change_rate': 0.0,
                    f'gsd_bsd/{prefix}_base_surrogate': 0.0,
                    f'gsd_bsd/{prefix}_target_surrogate': 0.0,
                    f'gsd_bsd/{prefix}_surrogate_improvement': 0.0,
                    f'gsd_bsd/{prefix}_target_changed_rate': 0.0,
                    f'gsd_bsd/{prefix}_target_search_candidates': 0.0,
                    f'gsd_bsd/{prefix}_target_search_time_ms': 0.0,
                    f'gsd_bsd/{prefix}_denoiser_time_ms': 0.0,
                    f'gsd_bsd/{prefix}_operation_loss': 0.0,
                    f'gsd_bsd/{prefix}_noop_loss': 0.0,
                }

            base_mask = bsd_aux.get(f'{prefix}_base_mask')
            hard_mask = bsd_aux.get(f'{prefix}_hard_mask')
            soft_mask = bsd_aux.get(f'{prefix}_soft_mask')
            valid_mask = bsd_aux.get(f'{prefix}_valid_mask')
            valid_ops = bsd_aux.get(f'{prefix}_operation_valid')
            if base_mask is None or hard_mask is None or valid_mask is None or valid_ops is None:
                zero = torch.tensor(0.0, device=operation_logits.device)
                return zero, {}

            base_mask = base_mask.to(device=operation_logits.device)
            hard_mask = hard_mask.to(device=operation_logits.device)
            soft_mask = soft_mask.to(device=operation_logits.device) if soft_mask is not None else base_mask
            valid_mask = valid_mask.to(device=operation_logits.device)
            valid_ops = valid_ops.to(device=operation_logits.device)
            masked_logits = operation_logits.masked_fill(~valid_ops, -1e9)
            valid_count = valid_mask.sum(dim=-1).float()
            selected_count = base_mask.sum(dim=-1).float()

            drop_mask = (base_mask > 0.5) & (hard_mask <= 0.5)
            add_mask = (base_mask <= 0.5) & (hard_mask > 0.5)
            noop = ~(drop_mask.any(dim=-1) & add_mask.any(dim=-1))
            drop_index = torch.full((base_mask.shape[0],), -1, dtype=torch.long, device=base_mask.device)
            add_index = torch.full((base_mask.shape[0],), -1, dtype=torch.long, device=base_mask.device)
            if drop_mask.numel() > 0:
                drop_index = torch.where(
                    drop_mask.any(dim=-1),
                    drop_mask.to(dtype=torch.long).argmax(dim=-1),
                    drop_index,
                )
                add_index = torch.where(
                    add_mask.any(dim=-1),
                    add_mask.to(dtype=torch.long).argmax(dim=-1),
                    add_index,
                )

            try:
                target_index = operation_target_index(operation, drop_index, add_index, noop)
            except Exception:
                target_index = torch.zeros(base_mask.shape[0], dtype=torch.long, device=base_mask.device)
                noop = torch.ones_like(noop)

            per_sample_loss = F.cross_entropy(masked_logits, target_index, reduction='none')
            noop_weight = torch.where(
                noop.to(dtype=torch.bool),
                torch.full_like(per_sample_loss, float(self.lambda_gsd_bsd_noop)),
                torch.ones_like(per_sample_loss),
            )
            op_loss = (per_sample_loss * noop_weight).mean()
            probs = F.softmax(masked_logits, dim=-1)
            valid_probs = probs * valid_ops.to(dtype=probs.dtype)
            valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            op_entropy = -(valid_probs * valid_probs.clamp_min(1e-12).log()).sum(dim=-1).mean()

            swap_rate = (~noop).float().mean().detach()
            noop_rate = noop.float().mean().detach()
            mask_change = (hard_mask - base_mask).abs().sum(dim=-1) / valid_count.clamp_min(1.0)
            soft_budget_error = (soft_mask.sum(dim=-1) - base_mask.sum(dim=-1)).abs().mean().detach()
            target_changed = (~noop).float().mean().detach()

            info = {
                f'gsd_bsd/{prefix}_diffusion_loss': op_loss.detach().item(),
                f'gsd_bsd/{prefix}_valid_count': valid_count.mean().detach().item(),
                f'gsd_bsd/{prefix}_selected_count': selected_count.mean().detach().item(),
                f'gsd_bsd/{prefix}_swap_rate': swap_rate.item(),
                f'gsd_bsd/{prefix}_noop_rate': noop_rate.item(),
                f'gsd_bsd/{prefix}_operation_entropy': op_entropy.detach().item(),
                f'gsd_bsd/{prefix}_soft_budget_error': soft_budget_error.item(),
                f'gsd_bsd/{prefix}_mask_change_rate': mask_change.mean().detach().item(),
                f'gsd_bsd/{prefix}_base_surrogate': 0.0,
                f'gsd_bsd/{prefix}_target_surrogate': 0.0,
                f'gsd_bsd/{prefix}_surrogate_improvement': 0.0,
                f'gsd_bsd/{prefix}_target_changed_rate': target_changed.item(),
                f'gsd_bsd/{prefix}_target_search_candidates': float(operation_logits.shape[-1]),
                f'gsd_bsd/{prefix}_target_search_time_ms': 0.0,
                f'gsd_bsd/{prefix}_denoiser_time_ms': 0.0,
                f'gsd_bsd/{prefix}_operation_loss': op_loss.detach().item(),
                f'gsd_bsd/{prefix}_noop_loss': (per_sample_loss[noop] * self.lambda_gsd_bsd_noop).mean().detach().item() if noop.any() else 0.0,
            }
            return op_loss, info

        ally_loss, ally_info = _branch_loss('ally')
        enemy_loss, enemy_info = _branch_loss('enemy')
        total = self.lambda_gsd_bsd_diff * (ally_loss + enemy_loss)
        if self.lambda_gsd_bsd_action_consistency:
            total = total + self.lambda_gsd_bsd_action_consistency * torch.tensor(0.0, device=total.device)

        info = {
            'gsd_bsd/ally_diffusion_loss': ally_info.get('gsd_bsd/ally_diffusion_loss', 0.0),
            'gsd_bsd/enemy_diffusion_loss': enemy_info.get('gsd_bsd/enemy_diffusion_loss', 0.0),
            'gsd_bsd/total_diffusion_loss': total.detach().item(),
        }
        info.update(ally_info)
        info.update(enemy_info)
        return total, info

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        smd_loss = torch.tensor(0.0, device=self.device)
        smd_info = {}
        gsd_bsd_loss = torch.tensor(0.0, device=self.device)
        gsd_bsd_info = {}
        if self._use_smd:
            values, action_log_probs, dist_entropy, smd_aux = self.policy.evaluate_actions_with_smd(
                                                                              share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
            recon_loss = torch.tensor(0.0, device=self.device)
            smd_loss, smd_info = self.compute_smd_aux_loss(smd_aux, adv_targ)
        elif self._use_gsd_bsd:
            values, action_log_probs, dist_entropy, bsd_aux = self.policy.evaluate_actions_with_gsd_bsd(
                                                                              share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
            recon_loss = torch.tensor(0.0, device=self.device)
            gsd_bsd_loss, gsd_bsd_info = self.compute_gsd_bsd_aux_loss(bsd_aux)
        elif self._use_esmg and self._use_am_filter:
            values, action_log_probs, dist_entropy, recon_loss = self.policy.evaluate_actions_with_recon(
                                                                              share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        else:
            values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
            recon_loss = torch.tensor(0.0)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            total_actor_loss = policy_loss - dist_entropy * self.entropy_coef + recon_loss * self.recon_loss_coef + smd_loss + gsd_bsd_loss
            total_actor_loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        smd_info.update(gsd_bsd_info)
        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, smd_loss + gsd_bsd_loss, smd_info

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['smd_loss'] = 0
        train_info['mask_diffusion_loss'] = 0
        train_info['mask_distill_loss'] = 0
        train_info['sparse_loss'] = 0
        train_info['smd_valid_ratio'] = 0
        train_info['smd_avg_degree'] = 0
        train_info['smd_edge_prob_mean'] = 0
        train_info['smd_edge_prob_max'] = 0
        train_info['smd_hard_edge_ratio'] = 0
        train_info['smd_pseudo_positive_ratio'] = 0
        train_info['gsd_bsd/ally_diffusion_loss'] = 0
        train_info['gsd_bsd/enemy_diffusion_loss'] = 0
        train_info['gsd_bsd/total_diffusion_loss'] = 0
        train_info['gsd_bsd/ally_valid_count'] = 0
        train_info['gsd_bsd/enemy_valid_count'] = 0
        train_info['gsd_bsd/ally_selected_count'] = 0
        train_info['gsd_bsd/enemy_selected_count'] = 0
        train_info['gsd_bsd/ally_swap_rate'] = 0
        train_info['gsd_bsd/enemy_swap_rate'] = 0
        train_info['gsd_bsd/ally_noop_rate'] = 0
        train_info['gsd_bsd/enemy_noop_rate'] = 0
        train_info['gsd_bsd/ally_operation_entropy'] = 0
        train_info['gsd_bsd/enemy_operation_entropy'] = 0
        train_info['gsd_bsd/ally_soft_budget_error'] = 0
        train_info['gsd_bsd/enemy_soft_budget_error'] = 0
        train_info['gsd_bsd/ally_mask_change_rate'] = 0
        train_info['gsd_bsd/enemy_mask_change_rate'] = 0
        train_info['gsd_bsd/base_surrogate'] = 0
        train_info['gsd_bsd/target_surrogate'] = 0
        train_info['gsd_bsd/surrogate_improvement'] = 0
        train_info['gsd_bsd/target_changed_rate'] = 0
        train_info['gsd_bsd/target_search_candidates'] = 0
        train_info['gsd_bsd/target_search_time_ms'] = 0
        train_info['gsd_bsd/denoiser_time_ms'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, smd_loss, smd_info \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                for key in (
                    'smd_loss',
                    'mask_diffusion_loss',
                    'mask_distill_loss',
                    'sparse_loss',
                    'smd_valid_ratio',
                    'smd_avg_degree',
                    'smd_edge_prob_mean',
                    'smd_edge_prob_max',
                    'smd_hard_edge_ratio',
                    'smd_pseudo_positive_ratio',
                    'gsd_bsd/ally_diffusion_loss',
                    'gsd_bsd/enemy_diffusion_loss',
                    'gsd_bsd/total_diffusion_loss',
                    'gsd_bsd/ally_valid_count',
                    'gsd_bsd/enemy_valid_count',
                    'gsd_bsd/ally_selected_count',
                    'gsd_bsd/enemy_selected_count',
                    'gsd_bsd/ally_swap_rate',
                    'gsd_bsd/enemy_swap_rate',
                    'gsd_bsd/ally_noop_rate',
                    'gsd_bsd/enemy_noop_rate',
                    'gsd_bsd/ally_operation_entropy',
                    'gsd_bsd/enemy_operation_entropy',
                    'gsd_bsd/ally_soft_budget_error',
                    'gsd_bsd/enemy_soft_budget_error',
                    'gsd_bsd/ally_mask_change_rate',
                    'gsd_bsd/enemy_mask_change_rate',
                    'gsd_bsd/base_surrogate',
                    'gsd_bsd/target_surrogate',
                    'gsd_bsd/surrogate_improvement',
                    'gsd_bsd/target_changed_rate',
                    'gsd_bsd/target_search_candidates',
                    'gsd_bsd/target_search_time_ms',
                    'gsd_bsd/denoiser_time_ms',
                ):
                    train_info[key] += float(smd_info.get(key, 0.0))

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
