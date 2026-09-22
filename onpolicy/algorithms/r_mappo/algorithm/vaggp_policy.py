"""
VAGGP_Policy  -  Policy wrapper for VA-GGP adversary agents
============================================================
Drop-in replacement for R_MAPPOPolicy when training adversary agents
in simple_world_comm with the Variational Autoregressive Guided Graph
Policy architecture.

Components:
  actor  : AdversaryVGBSMLearner   (decentralised, graph + VIB)
  critic : R_Critic                (standard centralised critic)
  guider : AdversaryAutoregressiveGuider  (centralised teacher, training-only)
"""
import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Critic
from onpolicy.algorithms.r_mappo.algorithm.adversary_vaggp_actor import (
    AdversaryVGBSMLearner,
    AdversaryAutoregressiveGuider,
)
from onpolicy.utils.util import update_linear_schedule


class VAGGP_Policy:
    """
    Policy class for VA-GGP adversary agents.

    :param args:            (argparse.Namespace)
    :param obs_space:       (gym.Space) local observation space
    :param cent_obs_space:  (gym.Space) centralised observation space
    :param act_space:       (gym.Space) action space
    :param device:          (torch.device)
    :param enable_guider:   (bool) whether to instantiate the autoregressive guider
    :param num_adversaries: (int) total number of adversaries (needed by guider)
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,
                 device=torch.device("cpu"),
                 enable_guider=True, num_adversaries=4, adv_agent_id=0):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # ---- Actor: VGBSM Learner (with agent-ID for symmetry breaking) ----
        self.actor = AdversaryVGBSMLearner(args, obs_space, act_space, device,
                                           adv_agent_id=adv_agent_id)

        # ---- Critic: standard R_Critic ----
        self.critic = R_Critic(args, cent_obs_space, device)

        # ---- Guider: Autoregressive (optional, training only) ----
        self.guider = None
        self._enable_guider = enable_guider
        if enable_guider:
            self.guider = AdversaryAutoregressiveGuider(
                args, cent_obs_space, act_space, num_adversaries, device)

        # ---- Optimisers ----
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr,
            eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr,
            eps=self.opti_eps, weight_decay=self.weight_decay)
        self.guider_optimizer = None
        if self.guider is not None:
            guider_lr = getattr(args, 'vaggp_guider_lr', self.lr)
            self.guider_optimizer = torch.optim.Adam(
                self.guider.parameters(), lr=guider_lr,
                eps=self.opti_eps, weight_decay=self.weight_decay)

    # -----------------------------------------------------------------
    #  Learning-rate decay
    # -----------------------------------------------------------------
    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        if self.guider_optimizer is not None:
            guider_lr = self.guider_optimizer.param_groups[0]['lr']
            update_linear_schedule(self.guider_optimizer, episode, episodes, guider_lr)

    # -----------------------------------------------------------------
    #  Rollout API  (same signature as R_MAPPOPolicy)
    # -----------------------------------------------------------------
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,
                    masks, available_actions=None, deterministic=False):
        """Compute actions + values for environment stepping."""
        actions, action_log_probs, rnn_states_actor, vaggp_info = \
            self.actor(obs, rnn_states_actor, masks,
                       available_actions, deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        # Return `vaggp_info` in the 6th slot (was `latent_beliefs` in base policy)
        return values, actions, action_log_probs, rnn_states_actor, \
            rnn_states_critic, vaggp_info

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor,
                         rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """Standard evaluate_actions for compatibility with base R_MAPPO trainer."""
        action_log_probs, dist_entropy, _info = \
            self.actor.evaluate_actions(obs, rnn_states_actor, action,
                                        masks, available_actions, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks,
            available_actions=None, deterministic=False):
        actions, _, rnn_states_actor, _ = \
            self.actor(obs, rnn_states_actor, masks,
                       available_actions, deterministic)
        return actions, rnn_states_actor

