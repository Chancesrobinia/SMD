import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch 
        --cuda
            by default True, will use GPU to train; or else will use CPU; 
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.
    
    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.
    
    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer. 
    
    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies. 
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards. 
        --use_valuenorm
            by default True, use running mean and std to normalize rewards. 
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs. 
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.
    
    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)
    
    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss 
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm 
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.  
        --huber_delta <float>
            coefficient of huber loss.  
    
    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)
    
    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate
    
    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.
    
    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.
    
    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.
    
    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='mappo', choices=["rmappo", "mappo", "happo", "hatrpo", "mat", "mat_dec"])

    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='marl', help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=True, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='StarCraft2', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
    
    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action='store_true', default=False)
    parser.add_argument("--share_actor", action='store_true', default=False)

    # add for online multi-task
    parser.add_argument("--train_maps", type=str, nargs='+', default=None)
    parser.add_argument("--eval_maps", type=str, nargs='+', default=None)

    # ESMG (Ego Sparse Multiplex Graph) parameters
    parser.add_argument("--self_dim", type=int, default=None,
                        help="Dimension of ego-agent observation for ESMG. If None, ESMG is disabled and original MLPBase is used.")
    parser.add_argument("--neighbor_dim", type=int, default=None,
                        help="Dimension of each neighbor observation for ESMG / HeteroGraph. First 2 dims should be (dx, dy).")
    parser.add_argument("--k_max", type=int, default=10,
                        help="Max neighbors after KNN physical truncation.")
    parser.add_argument("--top_k_filter", type=int, default=5,
                        help="Number of neighbors kept after AM value filter.")
    parser.add_argument("--num_graph_layers", type=int, default=2,
                        help="Number of parallel graph attention layers (multiplex) in ESMG.")
    parser.add_argument("--beta_intrinsic", type=float, default=0.01,
                        help="Coefficient for SimHash intrinsic exploration reward.")

    # ESMG ablation study switches
    parser.add_argument("--use_knn", action='store_false', default=True,
                        help="If set, disable KNN physical truncation (process all neighbors).")
    parser.add_argument("--use_am_filter", action='store_false', default=True,
                        help="If set, disable AM value filter and reconstruction loss.")
    parser.add_argument("--esmg_layers", type=int, default=None,
                        help="Override num_graph_layers for ablation. If 1, degrades to single-layer graph. If None, falls back to --num_graph_layers.")
    parser.add_argument("--use_simhash_exploration", action='store_false', default=True,
                        help="If set, disable SimHash intrinsic exploration reward.")
    parser.add_argument("--recon_loss_coef", type=float, default=0.01,
                        help="Coefficient for ESMG reconstruction loss.")

    # VA-GGP (Variational Autoregressive Guided Graph Policy) parameters
    parser.add_argument("--use_vaggp", action='store_true', default=False,
                        help="Enable VA-GGP architecture for adversary agents in simple_world_comm.")
    parser.add_argument("--vaggp_lambda_rl", type=float, default=1.0,
                        help="Weight for auxiliary RL (PPO-Clip) loss in VA-GGP composite loss.")
    parser.add_argument("--vaggp_beta_recon", type=float, default=0.1,
                        help="Weight for VIB reconstruction loss in VA-GGP.")
    parser.add_argument("--vaggp_beta_kl", type=float, default=0.01,
                        help="Weight for KL bottleneck loss in VA-GGP.")
    parser.add_argument("--vaggp_delta_clip", type=float, default=1.5,
                        help="Double-clip threshold delta for MAGPO distillation loss.")
    parser.add_argument("--vaggp_guider_lr", type=float, default=5e-4,
                        help="Learning rate for the autoregressive guider network.")
    parser.add_argument("--z_dim", type=int, default=None,
                        help="Latent dimension for VIB. If None, defaults to hidden_size // 2.")
    parser.add_argument("--num_forests", type=int, default=2,
                        help="Number of forests in simple_world_comm.")
    parser.add_argument("--dim_c", type=int, default=4,
                        help="Communication channel dimension in simple_world_comm.")
    parser.add_argument("--num_entities", type=int, default=5,
                        help="Total number of landmark entities (landmarks + food + forests).")

    # Heterogeneous Dual-Graph Fusion parameters
    parser.add_argument("--use_hetero_graph", action='store_true', default=False,
                        help="Use Heterogeneous Dual-Graph (L2A + A2A) Actor instead of ESMG / MLP.")
    parser.add_argument("--use_gated_fusion", action='store_true', default=False,
                        help="Use feature-level gated concatenation in HeteroGraph terminal fusion.")
    parser.add_argument("--use_graph_raw_obs_fusion", action='store_true', default=False,
                        help="Fuse the local flat SMAC observation with the three-hop graph feature before the actor RNN.")
    parser.add_argument("--use_parallel_ally_graph", action='store_true', default=False,
                        help="Use parallel AllyGraph + AllEntityGraph actor base under --use_hetero_graph.")
    parser.add_argument("--use_entity_enemy_first_graph", action='store_true', default=False,
                        help="Use entity/enemy -> ally -> entity/enemy serial graph under --use_hetero_graph.")
    parser.add_argument("--hetero_graph_adversary_only", action='store_true', default=False,
                        help="In separated simple_world_comm, apply HeteroGraph only to adversary agents; good agents use vanilla MAPPO actor.")
    parser.add_argument("--agent_state_dim", type=int, default=4,
                        help="Dimension of ego physical state (p_vel + p_pos). Default 4 for MPE.")
    parser.add_argument("--landmark_dim", type=int, default=2,
                        help="Per-landmark feature dimension (relative pos). Default 2.")
    parser.add_argument("--num_neighbors", type=int, default=None,
                        help="Number of neighbor agents. If None, inferred from num_agents - 1.")
    parser.add_argument("--use_smd", action='store_true', default=False,
                        help="Enable GSD-Sparse-SMD sparse mask diffusion auxiliary method.")
    parser.add_argument("--smd_candidate_top_k", type=int, default=3,
                        help="Spatial Top-K teammate candidates for SMD Hop3.")
    parser.add_argument("--smd_student_hidden_dim", type=int, default=128,
                        help="Hidden dimension for StudentSparseMaskHead.")
    parser.add_argument("--smd_edge_top_m", type=int, default=1,
                        help="Number of online student edges kept per agent when using Top-M hard mask.")
    parser.add_argument("--smd_edge_threshold", type=float, default=0.5,
                        help="Threshold for student hard mask when Top-M is disabled.")
    parser.add_argument("--smd_use_topm_mask", action='store_true', default=True,
                        help="Use Top-M hard mask for online student edge selection.")
    parser.add_argument("--smd_disable_topm_mask", action='store_false', dest="smd_use_topm_mask",
                        help="Use threshold hard mask instead of Top-M.")
    parser.add_argument("--smd_pseudo_label_mode", type=str, default="adv_score",
                        choices=["adv_score"], help="Pseudo communication label construction mode.")
    parser.add_argument("--smd_pseudo_top_m", type=int, default=1,
                        help="Top-M pseudo communication edges from advantage-weighted utility.")
    parser.add_argument("--smd_use_positive_adv_only", action='store_true', default=False,
                        help="Only construct positive pseudo edges for samples with positive advantage.")
    parser.add_argument("--use_smd_diffusion_teacher", action='store_true', default=True,
                        help="Enable Sparse Mask Diffusion teacher auxiliary loss.")
    parser.add_argument("--disable_smd_diffusion_teacher", action='store_false', dest="use_smd_diffusion_teacher",
                        help="Disable SMD diffusion teacher auxiliary loss.")
    parser.add_argument("--smd_num_diffusion_steps", type=int, default=50,
                        help="Number of diffusion steps for SMD mask teacher.")
    parser.add_argument("--smd_beta_start", type=float, default=1e-4,
                        help="SMD diffusion beta start.")
    parser.add_argument("--smd_beta_end", type=float, default=2e-2,
                        help="SMD diffusion beta end.")
    parser.add_argument("--smd_teacher_hidden_dim", type=int, default=128,
                        help="Hidden dimension for SparseMaskDiffusionTeacher.")
    parser.add_argument("--lambda_smd_diff", type=float, default=0.01,
                        help="Weight for SMD mask diffusion loss.")
    parser.add_argument("--lambda_smd_distill", type=float, default=0.05,
                        help="Weight for Student mask BCE distillation loss.")
    parser.add_argument("--lambda_smd_sparse", type=float, default=0.001,
                        help="Weight for average communication degree sparse loss.")
    parser.add_argument("--smd_target_degree", type=float, default=1.0,
                        help="Target average online communication degree.")
    parser.add_argument("--smd_online_sampling", action='store_true', default=False,
                        help="Forbidden/legacy switch. SMD online diffusion sampling is disabled.")
    parser.add_argument("--smd_debug_shapes", action='store_true', default=False,
                        help="Print SMD tensor shapes once.")
    parser.add_argument("--smd_log_interval", type=int, default=1000,
                        help="Reserved SMD logging interval.")
    parser.add_argument("--use_gsd_bsd", action='store_true', default=False,
                        help="Enable GSD-BSD budget-preserving subgraph diffusion.")
    parser.add_argument("--gsd_bsd_ally_candidate_k", type=int, default=4)
    parser.add_argument("--gsd_bsd_enemy_candidate_k", type=int, default=4)
    parser.add_argument("--gsd_bsd_ally_edge_m", type=int, default=2)
    parser.add_argument("--gsd_bsd_enemy_edge_m", type=int, default=1)
    parser.add_argument("--gsd_bsd_hidden_dim", type=int, default=64)
    parser.add_argument("--gsd_bsd_num_heads", type=int, default=2)
    parser.add_argument("--gsd_bsd_num_layers", type=int, default=1)
    parser.add_argument("--gsd_bsd_type_embedding_dim", type=int, default=8)
    parser.add_argument("--gsd_bsd_state_embedding_dim", type=int, default=8)
    parser.add_argument("--gsd_bsd_time_embedding_dim", type=int, default=16)
    parser.add_argument("--gsd_bsd_denoise_steps", type=int, default=1)
    parser.add_argument("--gsd_bsd_num_ally_swap_candidates", type=int, default=4)
    parser.add_argument("--gsd_bsd_num_enemy_swap_candidates", type=int, default=4)
    parser.add_argument("--gsd_bsd_num_joint_candidates", type=int, default=4)
    parser.add_argument("--gsd_bsd_target_source", type=str, default="ppo_surrogate",
                        choices=["ppo_surrogate", "base_graph"])
    parser.add_argument("--gsd_bsd_target_improvement_margin", type=float, default=0.0)
    parser.add_argument("--gsd_bsd_target_search_interval", type=int, default=1)
    parser.add_argument("--gsd_bsd_target_batch_fraction", type=float, default=1.0)
    parser.add_argument("--gsd_bsd_use_st_mask", action='store_true', default=True)
    parser.add_argument("--gsd_bsd_disable_st_mask", action='store_false', dest="gsd_bsd_use_st_mask")
    parser.add_argument("--gsd_bsd_warmup_steps", type=int, default=0)
    parser.add_argument("--gsd_bsd_update_interval", type=int, default=1)
    parser.add_argument("--lambda_gsd_bsd_diff", type=float, default=0.01)
    parser.add_argument("--lambda_gsd_bsd_noop", type=float, default=1.0)
    parser.add_argument("--lambda_gsd_bsd_action_consistency", type=float, default=0.0)
    parser.add_argument("--gsd_bsd_enemy_base_score_source", type=str, default="auto",
                        choices=["auto", "hop2", "learned", "distance"])
    parser.add_argument("--gsd_bsd_debug", action='store_true', default=False)

    return parser
