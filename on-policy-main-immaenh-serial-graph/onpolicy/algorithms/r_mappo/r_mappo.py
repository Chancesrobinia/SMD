import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

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
            total_actor_loss = policy_loss - dist_entropy * self.entropy_coef + recon_loss * self.recon_loss_coef + smd_loss
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

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, smd_loss, smd_info

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
