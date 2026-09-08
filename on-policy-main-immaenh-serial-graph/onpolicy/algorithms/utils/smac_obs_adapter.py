import torch


class SMACHeteroObservationAdapter:
    """Split classic SMAC local observations into HeteroGraph inputs."""

    def __init__(self, obs_space, debug_shapes=False):
        if obs_space.__class__.__name__ != "list" or len(obs_space) < 5:
            raise ValueError(
                "Classic SMAC HeteroGraph requires observation-space metadata "
                "[obs_dim, ally_shape, enemy_shape, move_shape, own_extra_shape]."
            )

        self.obs_dim = self._read_scalar(obs_space[0], "obs_dim")
        self.n_allies, self.ally_raw_dim = self._read_entity_shape(
            obs_space[1], "ally"
        )
        self.n_enemies, self.enemy_raw_dim = self._read_entity_shape(
            obs_space[2], "enemy"
        )
        move_rows, self.move_dim = self._read_entity_shape(
            obs_space[3], "move"
        )
        own_rows, self.own_extra_dim = self._read_entity_shape(
            obs_space[4], "own_extra"
        )
        if move_rows != 1 or own_rows != 1:
            raise ValueError(
                "Classic SMAC move and own-extra metadata must each have one row."
            )
        if self.ally_raw_dim < 4 or self.enemy_raw_dim < 4:
            raise ValueError(
                "Classic SMAC ally/enemy features need at least "
                "[flag, distance, relative_x, relative_y]."
            )

        metadata_dim = (
            self.n_allies * self.ally_raw_dim
            + self.n_enemies * self.enemy_raw_dim
            + self.move_dim
            + self.own_extra_dim
        )
        if metadata_dim != self.obs_dim:
            raise ValueError(
                "Classic SMAC observation metadata sums to {}, but obs_dim is {}."
                .format(metadata_dim, self.obs_dim)
            )

        self.agent_state_dim = self.move_dim + self.own_extra_dim
        self.debug_shapes = bool(debug_shapes)
        self._debug_printed = False

    @staticmethod
    def _read_scalar(value, name):
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise ValueError("Invalid classic SMAC {} metadata: {!r}".format(name, value))
        if result < 0:
            raise ValueError("Classic SMAC {} must be non-negative.".format(name))
        return result

    @classmethod
    def _read_entity_shape(cls, shape, name):
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(
                "Classic SMAC {} metadata must be [count, feature_dim]."
                .format(name)
            )
        count = cls._read_scalar(shape[0], "{} count".format(name))
        feature_dim = cls._read_scalar(shape[1], "{} feature_dim".format(name))
        return count, feature_dim

    @staticmethod
    def _canonicalize(raw):
        return torch.cat(
            [
                raw[..., 2:4],
                raw[..., 1:2],
                raw[..., 0:1],
                raw[..., 4:],
            ],
            dim=-1,
        )

    def __call__(self, obs):
        if obs.ndim != 2:
            raise ValueError(
                "Classic SMAC adapter expects [batch, obs_dim], got shape {}."
                .format(tuple(obs.shape))
            )
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                "Classic SMAC observation size {} does not match metadata obs_dim {}."
                .format(obs.shape[-1], self.obs_dim)
            )

        batch_size = obs.shape[0]
        idx = 0

        ally_size = self.n_allies * self.ally_raw_dim
        ally_raw = obs[:, idx:idx + ally_size].contiguous().view(
            batch_size, self.n_allies, self.ally_raw_dim
        )
        idx += ally_size

        enemy_size = self.n_enemies * self.enemy_raw_dim
        enemy_raw = obs[:, idx:idx + enemy_size].contiguous().view(
            batch_size, self.n_enemies, self.enemy_raw_dim
        )
        idx += enemy_size

        move_feats = obs[:, idx:idx + self.move_dim]
        idx += self.move_dim

        own_extra_feats = obs[:, idx:idx + self.own_extra_dim]
        idx += self.own_extra_dim

        if idx != obs.shape[-1]:
            raise ValueError(
                "Classic SMAC adapter parsed {} values from observation size {}."
                .format(idx, obs.shape[-1])
            )

        ally_graph = self._canonicalize(ally_raw)
        enemy_graph = self._canonicalize(enemy_raw)
        agent_state = torch.cat([move_feats, own_extra_feats], dim=-1)

        ally_valid = ally_raw[..., 0] > 0
        enemy_valid = enemy_raw.abs().sum(dim=-1) > 1e-6
        ally_mask = ~ally_valid
        enemy_mask = ~enemy_valid

        ctx_obs = obs.new_zeros((batch_size, 0, 1))
        ctx_mask = torch.zeros(
            (batch_size, 0), dtype=torch.bool, device=obs.device
        )

        if self.debug_shapes and not self._debug_printed:
            print("[SMAC Adapter]")
            print("obs                ", tuple(obs.shape))
            print("agent_state        ", tuple(agent_state.shape))
            print("ally_raw           ", tuple(ally_raw.shape))
            print("ally_graph         ", tuple(ally_graph.shape))
            print("enemy_raw          ", tuple(enemy_raw.shape))
            print("enemy_graph        ", tuple(enemy_graph.shape))
            print("ally_mask          ", tuple(ally_mask.shape))
            print("enemy_mask         ", tuple(enemy_mask.shape))
            self._debug_printed = True

        return (
            agent_state,
            ally_graph,
            enemy_graph,
            ctx_obs,
            ally_mask,
            enemy_mask,
            ctx_mask,
        )
