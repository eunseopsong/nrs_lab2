# SPDX-License-Identifier: BSD-3-Clause
"""
v21.0: Dual Tracking Reward (Joint + Cartesian Position)
---------------------------------------------------------
- Reward Type: Exponential kernel (no tanh)
- Goal: Joint-space + Cartesian-space 병렬 학습 지원

Joint Reward:
    - q₂ 안정화 + velocity damping + weighted bias correction
    - Target: get_hdf5_target_joints(env, horizon=8)
Position Reward:
    - 6DoF (x, y, z, roll, pitch, yaw)
    - Target: get_hdf5_target_positions(env, horizon=1)

Visualization:
    (1) joint_tracking_v21_epX.png
    (2) pos_tracking_v21_epX.png
    (3) r_pose_total_joint_v21_epX.png
    (4) r_pose_total_pos_v21_epX.png
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import torch, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import (
    quat_to_euler_xyz,
    get_hdf5_target_joints,
    get_hdf5_target_positions,
)

# -----------------------------------------------------------
# Global States
# -----------------------------------------------------------
_joint_tracking_history = []
_joint_reward_history = []
_position_tracking_history = []
_position_reward_history = []
_episode_counter_joint = 0
_episode_counter_position = 0


# -----------------------------------------------------------
# (1) Joint Tracking Reward
# -----------------------------------------------------------
def joint_tracking_reward(env: "ManagerBasedRLEnv"):
    """Joint-space tracking reward (exponential kernel)"""
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos[:, :6], robot.data.joint_vel[:, :6]
    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    D = q.shape[1]
    step = int(env.common_step_counter)

    # Horizon-based target
    fut = get_hdf5_target_joints(env, horizon=8)
    q_star_curr, q_star_next = fut[:, :D], fut[:, D:2*D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    # Errors
    e_q, e_qd = q - q_star_next, qd - qd_star

    # Weights & gains
    wj = torch.tensor([1, 2.0, 1, 4.0, 1, 1], device=q.device).unsqueeze(0)
    k_pos = torch.tensor([1.0, 8.0, 2.0, 6.0, 2.0, 2.0], device=q.device).unsqueeze(0)
    k_vel = torch.tensor([0.10, 0.40, 0.10, 0.40, 0.10, 0.10], device=q.device).unsqueeze(0)

    e_q2, e_qd2 = wj * (e_q ** 2), wj * (e_qd ** 2)
    r_pose_jointwise = torch.exp(-k_pos * e_q2)
    r_vel_jointwise = torch.exp(-k_vel * e_qd2)
    r_pose, r_vel = r_pose_jointwise.sum(dim=1), r_vel_jointwise.sum(dim=1)
    total = 0.9 * r_pose + 0.1 * r_vel

    # Console log every 10 steps
    if step % 10 == 0:
        print(f"[Joint Step {step}] mean(e_q)={torch.norm(e_q, dim=1).mean():.3f}, total={total.mean():.3f}")
        mean_e_q_abs = torch.mean(torch.abs(e_q), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_jointwise, dim=0).cpu().numpy()
        for j in range(D):
            print(f"  joint{j+1}: |mean(e_q)|={mean_e_q_abs[j]:.3f}, r_pose={mean_r_pose[j]:.3f}")

    # History
    _joint_tracking_history.append((step, q_star_next[0].cpu().numpy(), q[0].cpu().numpy()))
    _joint_reward_history.append((step, r_pose_jointwise[0].cpu().numpy()))

    # Visualization
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_joint(step)

    return total


# -----------------------------------------------------------
# (2) Position Tracking Reward (6DoF)
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """6D End-effector pose tracking reward with exponential kernel"""
    robot = env.scene["robot"]
    device = robot.data.body_pos_w.device
    step = int(env.common_step_counter)

    # EE pose
    link_idx = robot.find_bodies("wrist_3_link")[0]
    pos_w = robot.data.body_pos_w[:, link_idx, :].squeeze(1)
    quat_w = robot.data.body_quat_w[:, link_idx, :].squeeze(1)
    euler_w = quat_to_euler_xyz(quat_w)
    ee_pose = torch.cat([pos_w, euler_w], dim=1)  # (N,6)

    # Target pose
    target_pose = get_hdf5_target_positions(env, horizon=1)
    if target_pose.ndim == 2 and target_pose.shape[1] > 6:
        target_pose = target_pose[:, :6]
    elif target_pose.ndim == 3:
        target_pose = target_pose.squeeze(1)

    # Reward
    e_pose = ee_pose - target_pose
    w = torch.tensor([1.0, 1.0, 1.0, 0.3, 0.3, 0.3], device=device).unsqueeze(0)
    e_sq = (w * e_pose) ** 2
    k_pos = 3.0
    r_pose_axiswise = torch.exp(-k_pos * e_sq)
    reward = torch.mean(r_pose_axiswise, dim=1)

    # History
    global _position_tracking_history, _position_reward_history
    _position_tracking_history.append((step, target_pose[0].cpu().numpy(), ee_pose[0].cpu().numpy()))
    _position_reward_history.append((step, reward[0].item()))

    # -------- console log every 10 steps --------
    if step % 10 == 0:
        mean_e_pose = torch.mean(torch.abs(e_pose), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_axiswise, dim=0).cpu().numpy()
        print(f"[Position Step {step:>5}] mean(|e_pose|)={torch.norm(e_pose, dim=1).mean():.4f}, total_reward={reward.mean():.4f}")
        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        for i in range(6):
            print(
                f"  {labels[i]:<6} | current={ee_pose[0,i]:+9.4f} | "
                f"target={target_pose[0,i]:+9.4f} | "
                f"error={mean_e_pose[i]:+9.4f} | "
                f"r_axis={mean_r_pose[i]:.4f}"
            )


    # Visualization
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_position(step)

    return reward


# -----------------------------------------------------------
# (3) Visualization: Joint Tracking
# -----------------------------------------------------------
def save_episode_plots_joint(step: int):
    """Episode 단위 Joint-space 시각화"""
    global _joint_tracking_history, _joint_reward_history, _episode_counter_joint

    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_lab2/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_joint_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)
    colors = ["r","g","b","orange","purple","gray"]

    plt.figure(figsize=(10,6))
    for j in range(targets.shape[1]):
        plt.plot(targets[:,j],"--",color=colors[j],label=f"Target q{j+1}")
        plt.plot(currents[:,j],"-",color=colors[j],label=f"Current q{j+1}")
    plt.legend(); plt.grid(True)
    plt.title("Joint Tracking (v21)")
    plt.xlabel("Step"); plt.ylabel("Joint [rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_tracking_v21_ep{_episode_counter_joint+1}.png"))
    plt.close()

    r_steps, r_values = zip(*_joint_reward_history)
    r_values = np.vstack(r_values)
    total_reward = np.sum(r_values, axis=1)

    def smooth(y, window=50):
        if len(y) < window: return y
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    plt.figure(figsize=(10,6))
    for j in range(r_values.shape[1]):
        smooth_y = smooth(r_values[:,j])
        smooth_x = np.linspace(r_steps[0], r_steps[-1], len(smooth_y))
        plt.plot(smooth_x, smooth_y, color=colors[j], label=f"r_pose(q{j+1})")

    smooth_total = smooth(total_reward)
    smooth_x_total = np.linspace(r_steps[0], r_steps[-1], len(smooth_total))
    plt.plot(smooth_x_total, smooth_total, color="black", linewidth=2.5, label="Total Reward")
    plt.legend(); plt.grid(True)
    plt.title("Per-Joint Reward + Total Reward (v21)")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_joint_v21_ep{_episode_counter_joint+1}.png"))
    plt.close()

    _joint_tracking_history.clear()
    _joint_reward_history.clear()
    _episode_counter_joint += 1


# -----------------------------------------------------------
# (4) Visualization: Position Tracking
# -----------------------------------------------------------
def save_episode_plots_position(step: int):
    """Episode 단위 Cartesian-space (6D) 시각화"""
    global _position_tracking_history, _position_reward_history, _episode_counter_position

    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_lab2/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_position_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)
    labels = ["x","y","z","roll","pitch","yaw"]
    colors = ["r","g","b","orange","purple","gray"]

    plt.figure(figsize=(12,8))
    for j in range(6):
        plt.plot(targets[:,j],"--",color=colors[j],label=f"Target {labels[j]}")
        plt.plot(currents[:,j],"-",color=colors[j],label=f"Current {labels[j]}")
    plt.legend(ncol=3); plt.grid(True)
    plt.title("EE 6D Pose Tracking (v21)")
    plt.xlabel("Step"); plt.ylabel("Pose [m/rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pos_tracking_v21_ep{_episode_counter_position+1}.png"))
    plt.close()

    r_steps, r_values = zip(*_position_reward_history)
    r_values = np.array(r_values).flatten()

    def smooth(y, window=50):
        if len(y) < window: return y
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    smooth_y = smooth(r_values)
    smooth_x = np.linspace(r_steps[0], r_steps[-1], len(smooth_y))
    plt.figure(figsize=(10,5))
    plt.plot(smooth_x, smooth_y, "g", linewidth=2.5, label="r_pose(6D pose)")
    plt.legend(); plt.grid(True)
    plt.title("6D Pose Reward (v21)")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_pos_v21_ep{_episode_counter_position+1}.png"))
    plt.close()

    _position_tracking_history.clear()
    _position_reward_history.clear()
    _episode_counter_position += 1
