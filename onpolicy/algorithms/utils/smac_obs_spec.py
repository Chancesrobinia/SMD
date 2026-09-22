"""Structured description of a classic SMAC agent observation.

The specification is derived from the environment's own observation-space
metadata, never from a map-name lookup table. `StarCraft2_Env.get_obs_size()`
returns

    [total, [n_allies, ally_dim], [n_enemies, enemy_dim],
     [1, move_dim], [1, own_plus_id_plus_timestep_dim]]

and that structure is the single source of truth used here.
"""

import numpy as np


SUPPORTED_STRUCTURED_ENVS = ('StarCraft2',)


class SMACObsSpec(object):
    """Field layout of one SMAC observation frame.

    The classic StarCraft2 environment lays a frame out as

        [ally_feats, enemy_feats, move_feats, own_feats, (agent_id), (timestep)]

    `own_feat_dim` covers the trailing block as reported by the environment,
    which already folds in the optional agent-id and timestep entries. They are
    reported separately only when the environment exposes them unambiguously.
    """

    def __init__(self, total_dim, n_allies, ally_feat_dim, n_enemies,
                 enemy_feat_dim, move_feat_dim, own_feat_dim,
                 agent_id_dim=0, timestep_dim=0):
        self.total_dim = int(total_dim)
        self.n_allies = int(n_allies)
        self.ally_feat_dim = int(ally_feat_dim)
        self.n_enemies = int(n_enemies)
        self.enemy_feat_dim = int(enemy_feat_dim)
        self.move_feat_dim = int(move_feat_dim)
        self.own_feat_dim = int(own_feat_dim)
        self.agent_id_dim = int(agent_id_dim)
        self.timestep_dim = int(timestep_dim)
        self._validate()

    @property
    def ally_block_dim(self):
        return self.n_allies * self.ally_feat_dim

    @property
    def enemy_block_dim(self):
        return self.n_enemies * self.enemy_feat_dim

    @property
    def agent_state_dim(self):
        """Width of the ego feature vector fed to the graph query."""
        return self.move_feat_dim + self.own_feat_dim

    @property
    def ctx_dim(self):
        """Context nodes carry the movement-availability features."""
        return self.move_feat_dim

    @property
    def expected_total(self):
        return (self.ally_block_dim + self.enemy_block_dim
                + self.move_feat_dim + self.own_feat_dim)

    def _validate(self):
        for name in ('total_dim', 'move_feat_dim', 'own_feat_dim'):
            if getattr(self, name) <= 0:
                raise ValueError(
                    'SMACObsSpec requires a positive {}, got {}'.format(
                        name, getattr(self, name)))
        for name in ('n_allies', 'ally_feat_dim', 'n_enemies', 'enemy_feat_dim'):
            if getattr(self, name) < 0:
                raise ValueError(
                    'SMACObsSpec requires a non-negative {}, got {}'.format(
                        name, getattr(self, name)))
        if self.n_allies > 0 and self.ally_feat_dim == 0:
            raise ValueError('n_allies > 0 requires ally_feat_dim > 0')
        if self.n_enemies > 0 and self.enemy_feat_dim == 0:
            raise ValueError('n_enemies > 0 requires enemy_feat_dim > 0')
        if self.expected_total != self.total_dim:
            raise ValueError(
                'SMAC observation fields do not sum to the reported total: '
                'allies {}x{} + enemies {}x{} + move {} + own {} = {} != {}'
                .format(self.n_allies, self.ally_feat_dim, self.n_enemies,
                        self.enemy_feat_dim, self.move_feat_dim,
                        self.own_feat_dim, self.expected_total, self.total_dim))

    @classmethod
    def from_obs_space(cls, obs_space, env_name='StarCraft2'):
        """Build a spec from a structured SMAC observation space.

        `obs_space` is the per-agent entry produced by
        `StarCraft2_Env.get_obs_size()`; a plain Box carries no field layout and
        is therefore rejected rather than guessed at.
        """
        if env_name not in SUPPORTED_STRUCTURED_ENVS:
            raise NotImplementedError(
                'Structured HeteroGraph observation parsing has not been '
                'implemented for env_name={!r}. Only {} exposes the field '
                'layout required to build a SMACObsSpec; refusing to reuse the '
                'classic parser and guess dimensions.'.format(
                    env_name, ' / '.join(SUPPORTED_STRUCTURED_ENVS)))

        structure = getattr(obs_space, 'shape', obs_space)
        if not isinstance(structure, (list, tuple)) or len(structure) < 5:
            raise NotImplementedError(
                'SMAC HeteroGraph requires the structured observation space '
                'returned by StarCraft2_Env.get_obs_size(), i.e. '
                '[total, [n_allies, ally_dim], [n_enemies, enemy_dim], '
                '[1, move_dim], [1, own_dim]]. Got {!r}, which carries no '
                'field layout.'.format(structure))

        def pair(entry, label):
            if not isinstance(entry, (list, tuple, np.ndarray)) or len(entry) != 2:
                raise NotImplementedError(
                    'Malformed SMAC observation metadata for {}: expected a '
                    '[count, dim] pair, got {!r}'.format(label, entry))
            return int(entry[0]), int(entry[1])

        total = int(structure[0])
        n_allies, ally_dim = pair(structure[1], 'allies')
        n_enemies, enemy_dim = pair(structure[2], 'enemies')
        _, move_dim = pair(structure[3], 'move features')
        _, own_dim = pair(structure[4], 'own features')

        return cls(total_dim=total, n_allies=n_allies, ally_feat_dim=ally_dim,
                   n_enemies=n_enemies, enemy_feat_dim=enemy_dim,
                   move_feat_dim=move_dim, own_feat_dim=own_dim)

    def describe(self):
        return (
            'SMACObsSpec(total_dim={}, n_allies={}, ally_feat_dim={}, '
            'n_enemies={}, enemy_feat_dim={}, move_feat_dim={}, '
            'own_feat_dim={}, agent_state_dim={}, ctx_dim={})'.format(
                self.total_dim, self.n_allies, self.ally_feat_dim,
                self.n_enemies, self.enemy_feat_dim, self.move_feat_dim,
                self.own_feat_dim, self.agent_state_dim, self.ctx_dim))

    def __repr__(self):
        return self.describe()
