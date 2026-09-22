"""
VAGGP_Trainer  -  PPO trainer with 4-term composite loss for VA-GGP
====================================================================
Drop-in replacement for R_MAPPO when training adversary agents with
the Variational Autoregressive Guided Graph Policy architecture.

Composite loss (per mini-batch):
  L_total = L_distill  +  lambda_rl * L_rl  +  beta_recon * L_recon
            + beta_kl * L_kl_bottleneck  -  entropy_coef * H(pi)

where:
  L_rl       = standard PPO-Clip surrogate
  L_distill  = MAGPO double-clipped KL  (learner -> guider)
  L_recon    = MSE  decoder(z) vs. ground-truth prey relative positions
  L_kl       = KL( q(z|obs) || N(0,I) )
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from onpolicy.algorithms.r_mappo.algorithm.adversary_vaggp_actor import (
    VariationalBottleneck,
    double_clip_distillation_loss,
)


class VAGGP_Trainer:
    """
    PPO trainer extended with VA-GGP composite loss for adversary agents.

    Identical to R_MAPPO except ppo_update() computes 4 loss terms
    and jointly optimises learner + guider + critic.
    """

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # ---- standard PPO hyper-parameters ----
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

        # ---- VA-GGP specific hyper-parameters ----
        self.lambda_rl = getattr(args, 'vaggp_lambda_rl', 1.0)
        self.beta_recon = getattr(args, 'vaggp_beta_recon', 0.1)
        self.beta_kl = getattr(args, 'vaggp_beta_kl', 0.01)
        self.delta_clip = getattr(args, 'vaggp_delta_clip', 1.5)

        # ---- value normaliser ----
        assert not (self._use_popart and self._use_valuenorm)
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    # ==================================================================
    #  Value loss  (identical to R_MAPPO)
    # ==================================================================
    def cal_value_loss(self, values, value_preds_batch,
                       return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch
        ).clamp(-self.clip_param, self.clip_param)

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped)
            error_original = (
                self.value_normalizer.normalize(return_batch) - values)
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            vloss_c = huber_loss(error_clipped, self.huber_delta)
            vloss_o = huber_loss(error_original, self.huber_delta)
        else:
            vloss_c = mse_loss(error_clipped)
            vloss_o = mse_loss(error_original)

        value_loss = (torch.max(vloss_o, vloss_c)
                      if self._use_clipped_value_loss else vloss_o)

        if self._use_value_active_masks:
            value_loss = ((value_loss * active_masks_batch).sum()
                          / active_masks_batch.sum())
        else:
            value_loss = value_loss.mean()
        return value_loss

    # ==================================================================
    #  Composite PPO update  (the core of VA-GGP)
    # ==================================================================
    def ppo_update(self, sample, update_actor=True,
                   prev_adv_actions_onehot=None):
        """
        One gradient step on the 4-term composite loss.

        Args:
            sample:  standard PPO batch tuple from the buffer generator
            update_actor:  whether to update the learner actor
            prev_adv_actions_onehot:  list of [B, MAX_OH] tensors for the
                autoregressive guider context, or None (first adversary
                in the chain / guider disabled)

        Returns:
            (value_loss, critic_grad_norm, policy_loss, dist_entropy,
             actor_grad_norm, imp_weights, diag_dict)
        """
        # ---- unpack sample ----
        if len(sample) == 12:
            (share_obs_batch, obs_batch, rnn_states_batch,
             rnn_states_critic_batch, actions_batch, value_preds_batch,
             return_batch, masks_batch, active_masks_batch,
             old_action_log_probs_batch, adv_targ,
             available_actions_batch) = sample
        else:
            (share_obs_batch, obs_batch, rnn_states_batch,
             rnn_states_critic_batch, actions_batch, value_preds_batch,
             return_batch, masks_batch, active_masks_batch,
             old_action_log_probs_batch, adv_targ,
             available_actions_batch, _) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # ==============================================================
        #  Forward pass:  learner evaluate_actions  (returns vaggp_info)
        # ==============================================================
        action_log_probs, dist_entropy, vaggp_info = \
            self.policy.actor.evaluate_actions(
                obs_batch, rnn_states_batch, actions_batch,
                masks_batch, available_actions_batch, active_masks_batch)

        values, _ = self.policy.critic(
            check(share_obs_batch).to(**self.tpdv),
            check(rnn_states_critic_batch).to(**self.tpdv),
            check(masks_batch).to(**self.tpdv))

        # ==============================================================
        #  Loss 1:  Auxiliary RL  (standard PPO-Clip)
        # ==============================================================
        imp_weights = torch.exp(
            action_log_probs - old_action_log_probs_batch)       # [B, 1]

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            rl_loss = (
                (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                 * active_masks_batch).sum() / active_masks_batch.sum())
        else:
            rl_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # ==============================================================
        #  Loss 2:  MAGPO distillation  (double-clipped KL)
        # ==============================================================
        if (self.policy.guider is not None
                and prev_adv_actions_onehot is not None):
            # Guider produces reference probs (no gradient through guider)
            with torch.no_grad():
                guider_probs = self.policy.guider(
                    share_obs_batch, prev_adv_actions_onehot,
                    available_actions_batch)                       # [B, A]

            # Learner full log-probs (need grad for distillation)
            learner_logprobs = self.policy.actor.get_probs(
                obs_batch, rnn_states_batch, masks_batch,
                available_actions_batch).log()                    # [B, A]

            distill_loss = double_clip_distillation_loss(
                learner_logprobs, guider_probs, delta=self.delta_clip)
        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        # ==============================================================
        #  Loss 3:  VGBSM reconstruction
        #    decoder(z) should predict prey relative positions
        # ==============================================================
        decoded = vaggp_info['decoded']        # [B, num_good*2]
        target = vaggp_info['target']          # [B, num_good*2]
        recon_loss = F.mse_loss(decoded, target.detach())

        # ==============================================================
        #  Loss 4:  VGBSM KL bottleneck
        #    KL( q(z|obs) || N(0,I) )
        # ==============================================================
        kl_bottleneck = VariationalBottleneck.kl_divergence(
            vaggp_info['mu'], vaggp_info['logvar'])

        # ==============================================================
        #  Total actor loss  (4 terms + entropy bonus)
        # ==============================================================
        total_actor_loss = (
            distill_loss
            + self.lambda_rl * rl_loss
            + self.beta_recon * recon_loss
            + self.beta_kl * kl_bottleneck
            - self.entropy_coef * dist_entropy
        )

        # ---- Backward + step  (learner) ----
        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            total_actor_loss.backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        # ---- Guider supervised update  (maximise log-prob of buffered actions) ----
        if (self.policy.guider is not None
                and prev_adv_actions_onehot is not None):
            guider_lp, guider_ent = self.policy.guider.get_log_probs_and_entropy(
                share_obs_batch, prev_adv_actions_onehot,
                actions_batch, available_actions_batch, active_masks_batch)
            guider_loss = -guider_lp.mean() - 0.01 * guider_ent

            self.policy.guider_optimizer.zero_grad()
            guider_loss.backward()
            if self._use_max_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.policy.guider.parameters(), self.max_grad_norm)
            self.policy.guider_optimizer.step()

        # ---- Critic update ----
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        # ---- diagnostics dict ----
        diag = dict(
            recon_loss=recon_loss.item(),
            kl_bottleneck=kl_bottleneck.item(),
            distill_loss=(distill_loss.item()
                          if torch.is_tensor(distill_loss)
                          else distill_loss),
        )
        return (value_loss, critic_grad_norm, rl_loss, dist_entropy,
                actor_grad_norm, imp_weights, diag)

    # ==================================================================
    #  Step 4: Policy back-tracking (representation sync)
    # ==================================================================
    def backtrack_sync(self):
        """
        Copy compatible weights from the learner's graph-frontend
        fusion layers into the guider's global encoder, keeping the
        two representation spaces roughly aligned.
        """
        if self.policy.guider is None:
            return
        src = self.policy.actor.graph.fusion.state_dict()
        tgt = self.policy.guider.global_enc.state_dict()
        for key in tgt:
            if key in src and src[key].shape == tgt[key].shape:
                tgt[key].copy_(src[key])
        self.policy.guider.global_enc.load_state_dict(tgt)

    # ==================================================================
    #  Full training loop  (ppo_epoch x mini_batch)
    # ==================================================================
    def train(self, buffer, update_actor=True,
              all_adv_buffers=None, adv_order_idx=0,
              ar_prev_agents=None):
        """
        Full training cycle.

        Args:
            buffer:           this agent's SeparatedReplayBuffer
            update_actor:     whether to update the learner actor
            all_adv_buffers:  list of all adversary SeparatedReplayBuffers
                              (needed to build autoregressive context)
            adv_order_idx:    this agent's position in the adversary ordering
            ar_prev_agents:   (list[int] | None) shuffled list of adversary
                              agent-IDs that precede this agent in the
                              current AR ordering.  When provided, overrides
                              the fixed 0..adv_order_idx-1 ordering.
        """
        # ---- compute advantages ----
        if self._use_popart or self._use_valuenorm:
            advantages = (buffer.returns[:-1]
                          - self.value_normalizer.denormalize(
                              buffer.value_preds[:-1]))
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        adv_copy = advantages.copy()
        adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_adv = np.nanmean(adv_copy)
        std_adv = np.nanstd(adv_copy)
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        # ---- pre-compute flattened prev-agent one-hot arrays ----
        # Each is np.ndarray (T*N, MAX_OH) matching the buffer's flat ordering
        prev_oh_flat_list = None
        if all_adv_buffers is not None:
            # AR Order Shuffling: use the shuffled prev-agent list when provided
            if ar_prev_agents is not None and len(ar_prev_agents) > 0:
                prev_oh_flat_list = self._build_prev_onehot_from_list(
                    all_adv_buffers, ar_prev_agents)
            elif ar_prev_agents is None and adv_order_idx > 0:
                # Fallback to fixed order for backward compatibility
                prev_oh_flat_list = self._build_prev_onehot(
                    all_adv_buffers, adv_order_idx)

        # ---- training loop ----
        train_info = dict(
            value_loss=0, policy_loss=0, dist_entropy=0,
            actor_grad_norm=0, critic_grad_norm=0, ratio=0,
            recon_loss=0, kl_bottleneck=0, distill_loss=0)

        for _ in range(self.ppo_epoch):
            # Use custom indexed generator so prev_oh aligns with sample
            gen = self._indexed_ff_generator(
                buffer, advantages, self.num_mini_batch,
                prev_oh_flat_list)

            for sample, prev_oh_batch in gen:
                ctx = None
                if prev_oh_batch is not None:
                    ctx = [torch.from_numpy(t) for t in prev_oh_batch]

                result = self.ppo_update(
                    sample, update_actor,
                    prev_adv_actions_onehot=ctx)

                (value_loss, critic_grad_norm, policy_loss,
                 dist_entropy, actor_grad_norm, imp_weights, diag) = result

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['recon_loss'] += diag['recon_loss']
                train_info['kl_bottleneck'] += diag['kl_bottleneck']
                train_info['distill_loss'] += diag['distill_loss']

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates

        # Step 4: back-tracking sync after each training cycle
        self.backtrack_sync()
        return train_info

    # ==================================================================
    #  Helpers
    # ==================================================================
    @staticmethod
    def _build_prev_onehot(all_adv_buffers, adv_order_idx):
        """
        Build one-hot action arrays for adversary agents 0..adv_order_idx-1.

        Returns a list of np.ndarray [T*N, MAX_OH=9] (one per prev agent),
        where T=episode_length, N=n_rollout_threads.

        NOTE: buf.actions has shape (episode_length, N, act_dim) — NO +1,
              so we do NOT strip the last row.
        """
        MAX_OH = 5   # all adversaries are Discrete(5), no leader comm
        result = []
        for prev_idx in range(adv_order_idx):
            buf = all_adv_buffers[prev_idx]
            # buf.actions shape: (episode_length, n_threads, act_dim)
            acts = buf.actions                          # (T, N, act_dim)
            flat = acts.reshape(-1, acts.shape[-1])     # (T*N, act_dim)
            act_dim = flat.shape[-1]

            if act_dim == 1:
                # Discrete(5): single integer action -> one-hot(5)
                # (all adversaries are equal — no leader, all silent)
                oh = np.zeros((flat.shape[0], 5), dtype=np.float32)
                idx = flat[:, 0].astype(int).clip(0, 4)
                oh[np.arange(len(idx)), idx] = 1.0
            else:
                oh = np.zeros((flat.shape[0], MAX_OH), dtype=np.float32)

            # Pad to MAX_OH if needed
            if oh.shape[-1] < MAX_OH:
                oh = np.pad(oh, ((0, 0), (0, MAX_OH - oh.shape[-1])))

            result.append(oh)
        return result

    @staticmethod
    def _build_prev_onehot_from_list(all_adv_buffers, prev_agent_ids):
        """
        Build one-hot action arrays for a *shuffled* list of preceding agents.

        Unlike _build_prev_onehot (which always uses agents 0..idx-1),
        this accepts an arbitrary ordered list of agent IDs, enabling
        Autoregressive Order Shuffling to break the fixed decision order.

        Args:
            all_adv_buffers:  list of all adversary SeparatedReplayBuffers
            prev_agent_ids:   list[int] of adversary agent IDs in the
                              shuffled AR order that precede the current agent.
        Returns:
            list of np.ndarray [T*N, 5] (one per preceding agent), or None.
        """
        if not prev_agent_ids:
            return None
        MAX_OH = 5   # all adversaries are Discrete(5), no leader comm
        result = []
        for prev_idx in prev_agent_ids:
            buf = all_adv_buffers[prev_idx]
            acts = buf.actions                          # (T, N, act_dim)
            flat = acts.reshape(-1, acts.shape[-1])     # (T*N, act_dim)
            act_dim = flat.shape[-1]

            if act_dim == 1:
                oh = np.zeros((flat.shape[0], 5), dtype=np.float32)
                idx = flat[:, 0].astype(int).clip(0, 4)
                oh[np.arange(len(idx)), idx] = 1.0
            else:
                oh = np.zeros((flat.shape[0], MAX_OH), dtype=np.float32)

            if oh.shape[-1] < MAX_OH:
                oh = np.pad(oh, ((0, 0), (0, MAX_OH - oh.shape[-1])))

            result.append(oh)
        return result

    @staticmethod
    def _indexed_ff_generator(buffer, advantages, num_mini_batch,
                              prev_oh_flat_list=None):
        """
        Feed-forward mini-batch generator that uses the **same random
        indices** for both the replay-buffer data and the pre-computed
        prev-adversary one-hot arrays.

        Yields:
            (sample_tuple, prev_oh_batch)
            where prev_oh_batch is a list of np.ndarray [B, MAX_OH] or None.
        """
        episode_length, n_rollout_threads = buffer.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size]
                   for i in range(num_mini_batch)]

        # Flatten buffer arrays (same as SeparatedReplayBuffer.feed_forward_generator)
        share_obs = buffer.share_obs[:-1].reshape(-1, *buffer.share_obs.shape[2:])
        obs = buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:])
        rnn_states = buffer.rnn_states[:-1].reshape(-1, *buffer.rnn_states.shape[2:])
        rnn_states_critic = buffer.rnn_states_critic[:-1].reshape(
            -1, *buffer.rnn_states_critic.shape[2:])
        actions = buffer.actions.reshape(-1, buffer.actions.shape[-1])
        if buffer.available_actions is not None:
            available_actions = buffer.available_actions[:-1].reshape(
                -1, buffer.available_actions.shape[-1])
        value_preds = buffer.value_preds[:-1].reshape(-1, 1)
        returns = buffer.returns[:-1].reshape(-1, 1)
        masks = buffer.masks[:-1].reshape(-1, 1)
        active_masks = buffer.active_masks[:-1].reshape(-1, 1)
        action_log_probs = buffer.action_log_probs.reshape(
            -1, buffer.action_log_probs.shape[-1])
        if buffer.factor is not None:
            factor = buffer.factor.reshape(-1, buffer.factor.shape[-1])
        adv_flat = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            available_actions_batch = (available_actions[indices]
                                       if buffer.available_actions is not None
                                       else None)
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            adv_targ = adv_flat[indices]

            if buffer.factor is None:
                sample = (share_obs_batch, obs_batch, rnn_states_batch,
                          rnn_states_critic_batch, actions_batch,
                          value_preds_batch, return_batch, masks_batch,
                          active_masks_batch, old_action_log_probs_batch,
                          adv_targ, available_actions_batch)
            else:
                factor_batch = factor[indices]
                sample = (share_obs_batch, obs_batch, rnn_states_batch,
                          rnn_states_critic_batch, actions_batch,
                          value_preds_batch, return_batch, masks_batch,
                          active_masks_batch, old_action_log_probs_batch,
                          adv_targ, available_actions_batch, factor_batch)

            # Index prev_oh with same indices
            prev_oh_batch = None
            if prev_oh_flat_list is not None:
                prev_oh_batch = [arr[indices] for arr in prev_oh_flat_list]

            yield sample, prev_oh_batch

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        if self.policy.guider is not None:
            self.policy.guider.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        if self.policy.guider is not None:
            self.policy.guider.eval()

