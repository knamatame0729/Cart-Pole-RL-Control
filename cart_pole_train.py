import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from cart_pole_env import CartPoleEnv

def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.02,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0005,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 10,
            "num_mini_batches": 8,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }
    return train_cfg_dict

def get_cfgs():
    env_cfg = {
        "num_actions": 1,
        "episode_length_s": 20.0,
        "action_scale": 40.0,
        "simulate_action_latency": False,
        "clip_actions": 1.0,
        "base_init_pos": [0.0, 0.0, 0.0],  # Initial cart position
    }
    obs_cfg = {
        "num_obs": 4,  # Cart position, cart velocity, pole angle, pole angular velocity
        "obs_scales": {
            "cart_pos": 1.0,
            "cart_vel": 1.0,
            "pole_angle": 1.0,
            "pole_vel": 1.0,
        },
    }
    reward_cfg = {
        "angle_threshold": 0.2,
        "reward_scales": {
            "upright": 60.0,  # Reward for swinging pole upright
            "upright_stable": 40.0, # Reward for keeping upright
            "action_rate": -0.001,  # Penalty for rapid action changes
            "cart_pos": -2.0,  # Penalty for cart deviation from origin
        },
    }
    return env_cfg, obs_cfg, reward_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="cartpole-training")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    # Initialize genesis
    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Initialize CartPole environment
    env = CartPoleEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
        eval_mode=False,
    )

    # Initialize PPO runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Start training
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
