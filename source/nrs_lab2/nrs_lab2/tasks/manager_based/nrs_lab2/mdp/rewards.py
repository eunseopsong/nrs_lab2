# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions and tracking visualization for UR10e spindle environment.
"""

import torch
import matplotlib.pyplot as plt
import os
from isaaclab.envs import ManagerBasedRLEnv
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import get_hdf5_target

# ------------------------------------------------------
# Visualization output dir
# ------------------------------------------------------
_output_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
os.makedirs(_output_dir, exist_ok=True)


# ------------------------------------------------------
# Reward: joint tracking error
# ------------------------------------------------------
def joint_command_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Negative squared error between current and target joints."""
    joint_pos = env.scene["robot"].data.joint_pos
    env_ids = torch.arange(env.num_envs, device=env.device)
    target_pos = get_hdf5_target(env, env_ids)

    error = torch.sum((joint_pos - target_pos) ** 2, dim=1)
    reward = -error

    # Debug print
    print(f"[DEBUG] reward mean_error={error.mean().item():.6f}")

    return reward


def joint_command_error_tanh(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """Reward shaped with tanh on joint error."""
    joint_pos = env.scene["robot"].data.joint_pos
    env_ids = torch.arange(env.num_envs, device=env.device)
    target_pos = get_hdf5_target(env, env_ids)

    error = torch.sum((joint_pos - target_pos) ** 2, dim=1)
    reward = torch.tanh(-error / std)

    # Debug print
    print(f"[DEBUG] tanh reward mean_error={error.mean().item():.6f}")

    return reward


# ------------------------------------------------------
# Visualization every episode (30s default)
# ------------------------------------------------------
def visualize_tracking(env: ManagerBasedRLEnv):
    """Plot joint target vs actual at the end of an episode."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    targets = get_hdf5_target(env, env_ids)

    step = int(env.episode_length_buf[0].item())
    E = env.max_episode_length
    if step != E - 1:
        return  # Only visualize at end of episode

    dof = env.scene["robot"].num_joints
    joint_pos = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
    target_pos = targets[0].detach().cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(range(dof), joint_pos, "o-", label="Actual joints")
    plt.plot(range(dof), target_pos, "x--", label="Target joints")
    plt.xlabel("Joint index")
    plt.ylabel("Joint position [rad]")
    plt.title(f"Tracking at episode end (step={step})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(_output_dir, f"tracking_ep{env.episode_counter}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved tracking plot at {save_path}")
