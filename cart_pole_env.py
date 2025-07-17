import torch
import math
import genesis as gs

def gs_rand_float(lower, upper, shape, device):
    """Generate random floats in the range (lower, upper) with the given shape on the specified device."""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class CartPoleEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False, eval_mode=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]  # 4: cart position, cart velocity, pole angle, pole angular velocity
        self.num_actions = env_cfg["num_actions"]  # 1: force applied to cart
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", False)

        self.dt = 0.02 # time step
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.eval_mode = eval_mode # Eval mode flag: switch train mode and eval mode

        # Create simulation scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=10),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=30,
                camera_pos=(2.0, -2.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # Add ground plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # Add CartPole robot
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=gs.device) # Initial Position
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="cart_pole_v2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
            ),
        )

        # Build the simulation
        self.scene.build(n_envs=num_envs)

        # Define joint indices
        self.cart_joint_idx = [self.robot.get_joint("cart_to_base").dof_start] 
        self.pole_joint_idx = [self.robot.get_joint("pole_joint").dof_start]

        # Prepare reward functions and scale by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # Initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.cart_pos = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.cart_vel = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.pole_angle = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.pole_vel = torch.zeros((self.num_envs, 1), device=gs.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.extras = dict()
        self.extras["observations"] = dict()

    def step(self, actions):

        """
        Execute one step
        Actions to apply to the cart
        Return: obsercations, rewards, resets, extras 
        """

        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])  # Clipping actions to limit excessive actions
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        total_force = exec_actions * self.env_cfg["action_scale"]        # Actions → Scale
        self.robot.control_dofs_force(total_force, self.cart_joint_idx)  # Apply force to cart
        self.scene.step()                                                # +0.02 time step

        # Update buffers
        self.episode_length_buf += 1            # +1 episode step
        self.base_pos[:] = self.robot.get_pos() # Update base link position

        # Update buffers of joint states
        self.cart_pos[:] = self.robot.get_dofs_position(self.cart_joint_idx)
        self.cart_vel[:] = self.robot.get_dofs_velocity(self.cart_joint_idx)
        self.pole_angle[:] = self.robot.get_dofs_position(self.pole_joint_idx)
        self.pole_vel[:] = self.robot.get_dofs_velocity(self.pole_joint_idx)

        # Check termination conditions
        self.reset_buf = self.episode_length_buf >= self.max_episode_length  # 500 steps
        self.reset_buf |= torch.abs(self.cart_pos[:, 0]) > 2.4  # Cart position limit

        time_out_idx = (self.episode_length_buf >= self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # Reset 
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Compute rewards
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Compute observations
        self.obs_buf = torch.cat(
            [
                self.cart_pos * self.obs_scales["cart_pos"],
                self.cart_vel * self.obs_scales["cart_vel"],
                self.pole_angle * self.obs_scales["pole_angle"],
                self.pole_vel * self.obs_scales["pole_vel"],
            ],
            axis=-1,
        )
        
        # Hold current actions (when self.simulate_action_latency=True)
        self.last_actions[:] = self.actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        """Return current observations."""
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def reset_idx(self, envs_idx):
        """Reset specified environments"""
        if len(envs_idx) == 0:
            return

        # Reset joints
        self.cart_pos[envs_idx] = torch.zeros((len(envs_idx), 1), device=gs.device, dtype=gs.tc_float)
        self.cart_vel[envs_idx] = 0.0

        # Eval mode → Reset joint angle from downward position
        if self.eval_mode:
            self.pole_angle[envs_idx] = torch.zeros((len(envs_idx), 1), device=gs.device, dtype=gs.tc_float)
        # Train mode → Reset joint angle between -pi and pi
        else:
            self.pole_angle[envs_idx] = gs_rand_float(-3.14159, 3.14159, (len(envs_idx), 1), device=gs.device)
        
        self.pole_vel[envs_idx] = 0.0

        self.robot.set_dofs_position(
            position=self.cart_pos[envs_idx],
            dofs_idx_local=self.cart_joint_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            position=self.pole_angle[envs_idx],
            dofs_idx_local=self.pole_joint_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Log episode information
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        """Reset all environments."""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ Reward Functions ------------
    def _reward_upright(self):
        """
        Reward for swinging the pole upstraiht

        cos(pi) = -1, cos(0) = 1

        > 1 - cos(pi) = 2
        > 1 - cos(0) = 0

        """
        return (1.0 - torch.cos(self.pole_angle[:, 0])) / 1.0

    def _reward_upright_stable(self):
        """
        Reward for keeping the pole upright
        if the pole angle is between pi ± angle_threshold, +1 reward

        """
        upright_condition = torch.abs(self.pole_angle[:, 0] - 3.14159) < self.reward_cfg["angle_threshold"]
        return upright_condition.float() * 1.0

    def _reward_action_rate(self):
        """Penalty for large changes in actions (negative reward)
        
        """
        return -torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_cart_pos(self):
        """Penalty for cart deviation from origin (negative reward)."""
        return -torch.square(self.cart_pos[:, 0])