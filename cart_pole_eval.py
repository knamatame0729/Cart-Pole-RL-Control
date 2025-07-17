import argparse
import os
import pickle
from importlib import metadata

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

import torch

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

class CartPoleStatePublisher(Node):
    def __init__(self):
        super().__init__('cart_pole_state_publisher')
        """ Create publishers """
        self.cart_pos_pub = self.create_publisher(Float32, 'cart_pos', 10)
        self.cart_vel_pub = self.create_publisher(Float32, 'cart_vel', 10)
        self.pole_angle_pub = self.create_publisher(Float32, 'pole_angle', 10)
        self.pole_vel_pub = self.create_publisher(Float32, 'pole_vel', 10)
        self.action_pub = self.create_publisher(Float32, 'action', 10)

    def publish_state(self, cart_pos, cart_vel, pole_angle, pole_vel, action):
        """ Publish each state """
        cart_pos_msg = Float32()
        cart_pos_msg.data = float(cart_pos.item())
        self.cart_vel_pub.publish(cart_pos_msg)

        cart_vel_msg = Float32()
        cart_vel_msg.data = float(cart_vel.item())
        self.cart_vel_pub.publish(cart_vel_msg)

        pole_angle_msg = Float32()
        pole_angle_msg.data = float(pole_angle.item())
        self.pole_angle_pub.publish(pole_angle_msg)

        pole_vel_msg = Float32()
        pole_vel_msg.data = float(pole_vel.item())
        self.pole_vel_pub.publish(pole_vel_msg)
        
        action_msg = Float32()
        action_msg.data = float(action.item())
        self.action_pub.publish(action_msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="cartpole-training")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()
    
    # Initialize ros2
    rclpy.init()
    state_publisher = CartPoleStatePublisher()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))

    env = CartPoleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=True,
        eval_mode = True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)

            # Publish
            state_publisher.publish_state(env.cart_pos[0], env.cart_vel[0], env.pole_angle[0], env.pole_vel[0], actions[0])

            if dones.any():
                obs, _ = env.reset()
                print(f"Episode info: {infos['episode']}")

            rclpy.spin_once(state_publisher, timeout_sec=0.0)


if __name__ == "__main__":
    try:
        main()
    finally:
        rclpy.shutdown()
