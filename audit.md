# Audit: SMAC GSD-Sparse / HeteroGraph on branch `5-28`

Baseline branch: `5-28`
Baseline SHA: `7aff6d1fbbdda189d8d697372c77b4f846fd53ec` (matches the expected SHA)
Work branch: `fix-smac-gsd-sparse`

Every claim below was re-verified against source and against a live StarCraft II
environment (SC2 `Base75689`, maps present for `3m`, `1c3s5z`, `MMM2`).

## A. Hetero shell scripts pass CLI flags that are never registered — CONFIRMED

`onpolicy/scripts/train_smac_hetero_scripts/train_smac_3m.sh` passes
`--num_agents 3 --n_allies 2 --n_enemies 3 --ally_feat_dim 8 --enemy_feat_dim 8`.
None of those names exist in `onpolicy/config.py`.

`onpolicy/scripts/train/train_smac.py:123` uses `parser.parse_known_args(args)[0]`,
so the unknown half is dropped without any message. Observed leftovers:

```
['--map_name','3m','--num_agents','3','--n_allies','2','--n_enemies','3',
 '--ally_feat_dim','8','--enemy_feat_dim','8']
```

A misspelling such as `--ally_faet_dim 99` is likewise swallowed, so a typo
silently trains a differently-shaped model.

## B. R_Actor therefore falls back to wrong values — CONFIRMED

`num_agents` is unset, and `n_allies` / `n_enemies` fall back to defaults instead
of the map's real counts.

## C. Observation layout — CONFIRMED (this fork differs from upstream SMAC)

`StarCraft2_Env.py:1116-1131` concatenates:

```
ally_feats, enemy_feats, move_feats, own_feats, [agent_id], [timestep]
```

Note this is **ally-first**, not the upstream `move, enemy, ally, own` order, so
any parser copied from upstream SMAC is wrong here.

## D. `get_obs_size()` already returns full structure metadata — CONFIRMED

`StarCraft2_Env.py:1649` returns
`[total, [n_allies, ally_dim], [n_enemies, enemy_dim], [1, move_dim], [1, own+id+timestep]]`.

## E/F/G. Metadata unused; ally and enemy dims genuinely differ — CONFIRMED

Live values (`obs_agent_id=True`, `obs_last_action=True`, `obs_timestep_number=False`):

| map | total | ally | enemy | move | own+id | recompute |
|---|---|---|---|---|---|---|
| 3m | 64 | [2, 14] | [3, 5] | [1, 4] | [1, 17] | 64 OK |
| 1c3s5z | 310 | [8, 24] | [9, 9] | [1, 4] | [1, 33] | 310 OK |
| MMM2 | 370 | [9, 26] | [12, 8] | [1, 4] | [1, 36] | 370 OK |

`ally_dim != enemy_dim` on all three maps, and MMM2 has 12 enemies vs 9 allies.
`r_actor_critic.py:653-655` passes `ally_dim = enemy_dim = neighbor_dim`, which
cannot represent any of these.

Concrete misparse for 3m: the model reads ally `2x8=16`, enemy `2x8=16`,
agent_state `32`, whereas truth is ally `2x14=28`, enemy `3x5=15`,
agent_state `4+17=21`. Field boundaries do not line up at all, so the network
has been consuming shuffled features.

## H. Synergy Top-K has no gradient path — CONFIRMED

In `hetero_graph.py` Hop3, `synergy_mlp` produces scores that go through
`torch.topk` -> integer indices -> `gather` of raw ally features. Integer
indices are not differentiable, so PPO loss cannot reach `synergy_mlp`.

## Additional findings

- Script paths are inconsistent: 21 hetero scripts use `../../train/train_smac.py`
  while one uses `../train/train_smac.py`.
- All 22 hetero scripts use `algo="mappo"` while the SMAC baselines use `rmappo`,
  so the current comparison is confounded.
- Generic KNN assumes ally features `[0], [1]` are `dx, dy`; in SMAC they are
  `visible, normalized_distance` (`StarCraft2_Env.py:1071-1074`).
- Enemy feature `[0]` is attack availability, not visibility
  (`StarCraft2_Env.py:1043`), so it must not be used as a visibility mask.
