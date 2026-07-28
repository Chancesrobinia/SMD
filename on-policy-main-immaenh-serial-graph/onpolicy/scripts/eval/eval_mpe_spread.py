#!/usr/bin/env python
"""
Standalone evaluation script for ESMG-MAPPO models on MPE simple_spread.

This script loads a trained ESMG actor checkpoint and evaluates it in the
*original* MPE environment, reporting pure environment rewards only (no
intrinsic reward).  The result is directly comparable to vanilla MAPPO
evaluation numbers.

Usage (from onpolicy/scripts/):
    python eval/eval_mpe_spread.py \
        --model_dir results/MPE/simple_spread/rmappo/check/run13/models \
        --num_agents 10 --num_landmarks 10 \
        --self_dim 24 --neighbor_dim 4 --k_max 3 \
        --top_k_filter 5 --num_graph_layers 2 \
        --eval_episodes 100 --seed 1
"""

import sys, os, argparse, time
import numpy as np
import torch

# ---- make sure the project root is importable ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import DummyVecEnv


def make_eval_env(all_args, seed):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)])


def parse_args():
    # Reuse project-wide config so all ESMG / network flags are registered
    parser = get_config()

    # MPE-specific (not in get_config)
    parser.add_argument('--scenario_name', type=str, default='simple_spread')
    parser.add_argument('--num_landmarks', type=int, default=10)
    parser.add_argument('--num_agents', type=int, default=10)

    # --eval_episodes is already registered in get_config() (default=32)

    all_args = parser.parse_known_args(sys.argv[1:])[0]

    # Force flags needed for evaluation
    all_args.use_render = False
    all_args.n_rollout_threads = 1

    # Match training config for rmappo
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False

    return all_args


def _t2n(x):
    return x.detach().cpu().numpy()


def main():
    all_args = parse_args()

    assert all_args.model_dir is not None and all_args.model_dir != "", \
        "Must set --model_dir to the directory containing actor.pt"

    # ---- device ----
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(1)

    # ---- seed ----
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # ---- env ----
    envs = make_eval_env(all_args, all_args.seed)
    num_agents = all_args.num_agents

    # ---- build policy (same architecture as training) ----
    from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

    obs_space = envs.observation_space[0]
    share_obs_space = envs.share_observation_space[0] if all_args.use_centralized_V \
        else envs.observation_space[0]
    act_space = envs.action_space[0]

    policy = Policy(all_args, obs_space, share_obs_space, act_space, device=device)

    # ---- load actor weights ----
    actor_path = os.path.join(str(all_args.model_dir), 'actor.pt')
    assert os.path.isfile(actor_path), f"actor.pt not found at {actor_path}"
    state_dict = torch.load(actor_path, map_location=device)
    policy.actor.load_state_dict(state_dict)
    policy.actor.eval()
    print(f"[Eval] Loaded actor from {actor_path}")

    # ---- evaluate ----
    all_episode_rewards = []

    for ep in range(all_args.eval_episodes):
        obs = envs.reset()
        rnn_states = np.zeros((1, num_agents, all_args.recurrent_N,
                               all_args.hidden_size), dtype=np.float32)
        masks = np.ones((1, num_agents, 1), dtype=np.float32)

        episode_reward = np.zeros((1, num_agents, 1), dtype=np.float32)

        for step in range(all_args.episode_length):
            with torch.no_grad():
                # policy.act() returns (actions, rnn_states)
                # Internally it calls actor.forward() which handles ESMG obs
                # splitting automatically. No intrinsic reward involved.
                action, rnn_states_out = policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True
                )

            actions = np.array(np.split(_t2n(action), 1))
            rnn_states = np.array(np.split(_t2n(rnn_states_out), 1))

            # Convert to env actions
            if envs.action_space[0].__class__.__name__ == 'Discrete':
                actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
            elif envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                actions_env = None
                for i in range(envs.action_space[0].shape):
                    uc = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                    actions_env = uc if actions_env is None else \
                        np.concatenate((actions_env, uc), axis=2)
            else:
                raise NotImplementedError

            # Step env — pure environment reward, no intrinsic bonus
            obs, rewards, dones, infos = envs.step(actions_env)
            episode_reward += rewards

            rnn_states[dones == True] = np.zeros(
                ((dones == True).sum(), all_args.recurrent_N,
                 all_args.hidden_size), dtype=np.float32)
            masks = np.ones((1, num_agents, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1),
                                            dtype=np.float32)

        # episode_reward: [1, num_agents, 1]
        # For shared reward (collaborative), all agents have the same reward
        mean_ep_reward = np.mean(episode_reward)
        all_episode_rewards.append(mean_ep_reward)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{all_args.eval_episodes}: "
                  f"reward = {mean_ep_reward:.4f}")

    envs.close()

    # ---- summary ----
    rewards_arr = np.array(all_episode_rewards)
    print("\n" + "=" * 60)
    print(f"Evaluation over {all_args.eval_episodes} episodes:")
    print(f"  Mean reward : {rewards_arr.mean():.4f}")
    print(f"  Std reward  : {rewards_arr.std():.4f}")
    print(f"  Min reward  : {rewards_arr.min():.4f}")
    print(f"  Max reward  : {rewards_arr.max():.4f}")
    print("=" * 60)
    print(f"\nThis number is directly comparable to vanilla MAPPO's "
          f"evaluation reward on the same environment config "
          f"(num_agents={num_agents}, num_landmarks={all_args.num_landmarks}, "
          f"episode_length={all_args.episode_length}).")


if __name__ == "__main__":
    main()
