"""
VA-GGP: Variational Autoregressive Guided Graph Policy
=======================================================
Network definitions for adversary agents in MPE simple_world_comm.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.utils.util import get_shape_from_obs_space
class GraphAttentionHop(nn.Module):
    """Single-hop scaled dot-product graph attention."""
    def __init__(self, query_dim, key_dim, hidden_size, use_orthogonal=True):
        super().__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def _init(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.W_q = _init(nn.Linear(query_dim, hidden_size))
        self.W_k = _init(nn.Linear(key_dim, hidden_size))
        self.W_v = _init(nn.Linear(key_dim, hidden_size))
        self.scale = math.sqrt(float(hidden_size))
        self._out_dim = hidden_size
    def forward(self, query, keys):
        B, K, _ = keys.shape
        if K == 0:
            return query.new_zeros(B, self._out_dim)
        q = self.W_q(query).unsqueeze(1)
        k = self.W_k(keys)
        v = self.W_v(keys)
        attn = (q * k).sum(-1) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn.unsqueeze(-1) * v).sum(dim=1)
        return out
class ThreeHopGraphFrontend(nn.Module):
    """
    3-hop relational graph for adversary in simple_world_comm.
    Hop 1: Self -> Teammate Adversaries (wolf-pack formation)
    Hop 2: Self+Hop1 -> Target Prey (lock-on good agents)
    Hop 3: Self+Hop2 -> Forests (terrain awareness)
    """
    def __init__(self, self_dim, adv_feat_dim, prey_feat_dim,
                 forest_feat_dim, entity_flat_dim, prey_forest_dim,
                 hidden_size, use_orthogonal=True):
        super().__init__()
        H = hidden_size
        self.hop1 = GraphAttentionHop(self_dim, adv_feat_dim, H, use_orthogonal)
        self.hop2 = GraphAttentionHop(self_dim + H, prey_feat_dim, H, use_orthogonal)
        self.hop3 = GraphAttentionHop(self_dim + H, forest_feat_dim, H, use_orthogonal)
        fusion_in = self_dim + 3 * H + entity_flat_dim + prey_forest_dim
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain('relu')
        def _init(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.fusion = nn.Sequential(
            _init(nn.Linear(fusion_in, H)), nn.ReLU(), nn.LayerNorm(H),
            _init(nn.Linear(H, H)),         nn.ReLU(), nn.LayerNorm(H))
    def forward(self, self_state, adv_teammate, target_prey,
                forest_feat, entity_flat, prey_forest):
        h1 = self.hop1(self_state, adv_teammate)
        h2 = self.hop2(torch.cat([self_state, h1], -1), target_prey)
        h3 = self.hop3(torch.cat([self_state, h2], -1), forest_feat)
        return self.fusion(
            torch.cat([self_state, h1, h2, h3, entity_flat, prey_forest], -1))
class VariationalBottleneck(nn.Module):
    """
    Variational Information Bottleneck.
    Safety: logvar clamped to [-20, +2], std always finite.
    """
    LOGVAR_LO = -20.0
    LOGVAR_HI = 2.0
    def __init__(self, input_dim, z_dim, decode_target_dim,
                 hidden_size, use_orthogonal=True):
        super().__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def _init(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.fc_mu = _init(nn.Linear(input_dim, z_dim))
        self.fc_logvar = _init(nn.Linear(input_dim, z_dim))
        self.decoder = nn.Sequential(
            _init(nn.Linear(z_dim, hidden_size)), nn.ReLU(),
            _init(nn.Linear(hidden_size, decode_target_dim)))
        self.z_dim = z_dim
    def encode(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x).clamp(self.LOGVAR_LO, self.LOGVAR_HI)
        return mu, logvar
    def reparameterise(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)
    def decode(self, z):
        return self.decoder(z)
    def forward(self, spatial_features, deterministic=False):
        mu, logvar = self.encode(spatial_features)
        z = self.reparameterise(mu, logvar, deterministic)
        decoded = self.decode(z)
        return z, mu, logvar, decoded
    @staticmethod
    def kl_divergence(mu, logvar):
        """KL( q(z|x) || N(0,I) ), averaged over batch."""
        return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
class AdversaryVGBSMLearner(nn.Module):
    """
    Decentralised actor for adversary agents in simple_world_comm.
    Pipeline: flat obs -> parse -> inject agent_id -> 3-Hop Graph -> VIB -> z_proj -> RNN -> ACTLayer
    Adversary obs layout (default 4 adv + 2 good, dim = 32):
      [p_vel(2)|p_pos(2)|entity_pos(5*2)|other_pos(5*2)|good_vel(2*2)|in_forest(2)|prey_forest(2)]

    NOTE: An agent-ID one-hot vector is concatenated to self_state BEFORE
          entering the graph, so that each adversary can learn a distinct
          role even under shared team reward.  This is the key mechanism
          that breaks policy symmetry among adversaries.
    """
    SELF_DIM = 4
    ENT_DIM = 2
    POS_DIM = 2
    VEL_DIM = 2
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu"),
                 adv_agent_id=0):
        super().__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        H = args.hidden_size
        self.hidden_size = H
        self.num_adv = getattr(args, 'num_adversaries', 4)
        self.num_good = getattr(args, 'num_good_agents', 2)
        self.num_ent = getattr(args, 'num_entities', 5)
        self.num_forests = getattr(args, 'num_forests', 2)
        self.prey_forest_dim = self.num_good          # one flag per prey
        self.num_adv_tm = self.num_adv - 1
        self.num_others = self.num_adv + self.num_good - 1
        n_non_forest = self.num_ent - self.num_forests
        adv_feat_dim = self.POS_DIM
        prey_feat_dim = self.POS_DIM + self.VEL_DIM
        forest_feat_dim = self.ENT_DIM + 1
        entity_flat_dim = n_non_forest * self.ENT_DIM
        self.z_dim = getattr(args, 'z_dim', None) or (H // 2)
        decode_tgt_dim = self.num_good * self.POS_DIM
        # ---- Agent ID: break symmetry among adversaries ----
        self._adv_agent_id = adv_agent_id
        self_dim_with_id = self.SELF_DIM + self.num_adv   # 4 + 4 = 8
        self.graph = ThreeHopGraphFrontend(
            self_dim_with_id, adv_feat_dim, prey_feat_dim, forest_feat_dim,
            entity_flat_dim, self.prey_forest_dim, H, getattr(args, 'use_orthogonal', True))
        self.vib = VariationalBottleneck(
            H, self.z_dim, decode_tgt_dim, H, getattr(args, 'use_orthogonal', True))
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            getattr(args, 'use_orthogonal', True)]
        gain = nn.init.calculate_gain('relu')
        def _init(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.z_proj = nn.Sequential(
            _init(nn.Linear(self.z_dim, H)), nn.ReLU(), nn.LayerNorm(H))
        self._use_rnn = getattr(args, 'use_recurrent_policy', True)
        self._use_naive_rnn = getattr(args, 'use_naive_recurrent_policy', False)
        self._recurrent_N = getattr(args, 'recurrent_N', 1)
        if self._use_rnn or self._use_naive_rnn:
            self.rnn = RNNLayer(H, H, self._recurrent_N,
                                getattr(args, 'use_orthogonal', True))
        self.act = ACTLayer(action_space, H,
                            getattr(args, 'use_orthogonal', True),
                            getattr(args, 'gain', 0.01), args)
        self._use_policy_active_masks = getattr(args, 'use_policy_active_masks', True)
        self.to(device)
    def _parse_obs(self, obs):
        B = obs.shape[0]
        idx = 0
        self_state = obs[:, idx:idx + self.SELF_DIM]; idx += self.SELF_DIM
        ent_total = self.num_ent * self.ENT_DIM
        ent_flat = obs[:, idx:idx + ent_total]
        ent_all = ent_flat.view(B, self.num_ent, self.ENT_DIM); idx += ent_total
        n_nf = self.num_ent - self.num_forests
        entity_flat = ent_flat[:, :n_nf * self.ENT_DIM]
        forest_rel_pos = ent_all[:, n_nf:, :]
        oth_total = self.num_others * self.POS_DIM
        other_pos = obs[:, idx:idx + oth_total].view(B, self.num_others, self.POS_DIM)
        idx += oth_total
        adv_tm_pos = other_pos[:, :self.num_adv_tm, :]
        good_rel_pos = other_pos[:, self.num_adv_tm:, :]
        vel_total = self.num_good * self.VEL_DIM
        good_vel = obs[:, idx:idx + vel_total].view(B, self.num_good, self.VEL_DIM)
        idx += vel_total
        in_forest = obs[:, idx:idx + self.num_forests]; idx += self.num_forests
        prey_forest = obs[:, idx:idx + self.prey_forest_dim]; idx += self.prey_forest_dim
        target_prey = torch.cat([good_rel_pos, good_vel], dim=-1)
        forest_feat = torch.cat(
            [forest_rel_pos, in_forest.view(B, self.num_forests, 1)], dim=-1)
        return dict(self_state=self_state, adv_teammate_pos=adv_tm_pos,
                    target_prey=target_prey, forest_feat=forest_feat,
                    entity_flat=entity_flat, prey_forest=prey_forest,
                    good_rel_pos=good_rel_pos.reshape(B, -1))
    def _extract(self, obs, deterministic=False):
        p = self._parse_obs(obs)
        B = obs.shape[0]
        # ---- inject agent ID one-hot to break symmetry ----
        agent_id_oh = obs.new_zeros(B, self.num_adv)
        agent_id_oh[:, self._adv_agent_id] = 1.0
        self_state_id = torch.cat([p['self_state'], agent_id_oh], dim=-1)
        spatial = self.graph(self_state_id, p['adv_teammate_pos'],
                             p['target_prey'], p['forest_feat'],
                             p['entity_flat'], p['prey_forest'])
        z, mu, lv, dec = self.vib(spatial, deterministic)
        feat = self.z_proj(z)
        return feat, z, mu, lv, dec, p['good_rel_pos']
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        feat, z, mu, lv, dec, tgt = self._extract(obs, deterministic)
        if self._use_rnn or self._use_naive_rnn:
            feat, rnn_states = self.rnn(feat, rnn_states, masks)
        actions, log_probs = self.act(feat, available_actions, deterministic)
        info = dict(z=z, mu=mu, logvar=lv, decoded=dec, target=tgt)
        return actions, log_probs, rnn_states, info
    def evaluate_actions(self, obs, rnn_states, action, masks,
                         available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        feat, z, mu, lv, dec, tgt = self._extract(obs, deterministic=False)
        if self._use_rnn or self._use_naive_rnn:
            feat, rnn_states = self.rnn(feat, rnn_states, masks)
        am = active_masks if self._use_policy_active_masks else None
        lp, ent = self.act.evaluate_actions(feat, action, available_actions, active_masks=am)
        info = dict(z=z, mu=mu, logvar=lv, decoded=dec, target=tgt)
        return lp, ent, info
    def get_probs(self, obs, rnn_states, masks, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        feat, *_ = self._extract(obs, deterministic=True)
        if self._use_rnn or self._use_naive_rnn:
            feat, _ = self.rnn(feat, rnn_states, masks)
        return self.act.get_probs(feat, available_actions)
class AdversaryAutoregressiveGuider(nn.Module):
    """
    God's-eye centralised teacher (training only).
    global_state -> MLP -> global_feat; prev_actions -> GRU -> ar_context
    fusion(global_feat, ar_context) -> ACTLayer -> reference distribution
    """
    MAX_OH = 5   # all adversaries are Discrete(5), no leader comm
    def __init__(self, args, share_obs_space, action_space,
                 num_adversaries, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        H = args.hidden_size
        self.hidden_size = H
        self.num_adv = num_adversaries
        s_shape = get_shape_from_obs_space(share_obs_space)
        s_dim = s_shape[0]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            getattr(args, 'use_orthogonal', True)]
        gain = nn.init.calculate_gain('relu')
        def _init(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.global_enc = nn.Sequential(
            _init(nn.Linear(s_dim, H)), nn.ReLU(), nn.LayerNorm(H),
            _init(nn.Linear(H, H)),     nn.ReLU(), nn.LayerNorm(H))
        self.act_embed = nn.Sequential(
            _init(nn.Linear(self.MAX_OH, H // 2)), nn.ReLU())
        self.ar_gru = nn.GRUCell(H // 2, H)
        self.fuse = nn.Sequential(
            _init(nn.Linear(H * 2, H)), nn.ReLU(), nn.LayerNorm(H))
        self.act = ACTLayer(action_space, H,
                            getattr(args, 'use_orthogonal', True),
                            getattr(args, 'gain', 0.01), args)
        self.to(device)
    def _ar_context(self, prev_oh_list, B):
        if len(prev_oh_list) == 0:
            return torch.zeros(B, self.hidden_size, device=self.device)
        h = torch.zeros(B, self.hidden_size, device=self.device)
        for oh in prev_oh_list:
            oh = check(oh).to(**self.tpdv)
            h = self.ar_gru(self.act_embed(oh), h)
        return h
    def forward(self, share_obs, prev_oh_list, available_actions=None):
        share_obs = check(share_obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        B = share_obs.shape[0]
        g = self.global_enc(share_obs)
        c = self._ar_context(prev_oh_list, B)
        f = self.fuse(torch.cat([g, c], dim=-1))
        return self.act.get_probs(f, available_actions)
    def get_log_probs_and_entropy(self, share_obs, prev_oh_list, actions,
                                  available_actions=None, active_masks=None):
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        B = share_obs.shape[0]
        g = self.global_enc(share_obs)
        c = self._ar_context(prev_oh_list, B)
        f = self.fuse(torch.cat([g, c], dim=-1))
        lp, ent = self.act.evaluate_actions(f, actions, available_actions, active_masks)
        return lp, ent
def double_clip_distillation_loss(learner_logprobs, guider_probs, delta=1.5):
    """
    MAGPO double-clipped KL distillation loss.
    ratio_a = exp(log p_L - log p_G); trigger when outside [1/delta, delta].
    loss = mean_B sum_a trigger_a * p_G^a * (log p_G^a - log p_L^a)
    """
    gp = guider_probs.detach().clamp(min=1e-8)
    gl = gp.log()
    ratio = (learner_logprobs - gl).exp()
    trigger = ((ratio > delta) | (ratio < 1.0 / delta)).float()
    kl_per_dim = gp * (gl - learner_logprobs)
    return (trigger * kl_per_dim).sum(dim=-1).mean()
