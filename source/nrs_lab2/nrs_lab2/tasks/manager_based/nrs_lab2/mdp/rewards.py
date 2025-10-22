# SPDX-License-Identifier: BSD-3-Clause
"""
v22.0: Dual Tracking Reward (Joint + Cartesian Position, FKSolver 기반)
----------------------------------------------------------------------
- Reward Type: Exponential kernel (no tanh)
- Goal: Joint-space + Cartesian-space 병렬 학습 (6DoF End-Effector 포함)

Changes from v21:
    ✅ version 변수를 통한 자동 title/file명 관리
    ✅ 주석 및 구조 정리
    ✅ 코드 안정화 및 시각화 일관성 유지

Joint Reward:
    - q₂ 안정화 + velocity damping + weighted bias correction
    - Target: get_hdf5_target_joints(env, horizon=8)

Position Reward:
    - 6DoF (x, y, z, roll, pitch, yaw)
    - Target: get_hdf5_target_positions(env, horizon=1)
    - FKSolver 기반 정확한 Forward Kinematics 사용

Visualization:
    (1) joint_tracking_<version>_epX.png
    (2) pos_tracking_<version>_epX.png
    (3) r_pose_total_joint_<version>_epX.png
    (4) r_pose_total_pos_<version>_epX.png
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
    get_hdf5_target_joints,
    get_hdf5_target_positions,
)

# -----------------------------------------------------------
# Global States
# -----------------------------------------------------------
version = "v22"  # ✅ 버전 관리 변수 (title / filename 자동 반영)

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

    # Visualization per episode
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_joint(step)

    return total


# -----------------------------------------------------------
# (2) Position Tracking Reward (6DoF, FKSolver 기반)
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """6D End-effector pose tracking reward (FKSolver 기반, exponential kernel)"""
    from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import get_ee_pose

    device = env.device
    step = int(env.common_step_counter)

    # (1) EE pose from FKSolver
    ee_pose = get_ee_pose(env)  # (N, 6): [x, y, z, roll, pitch, yaw]

    # (2) Target pose from HDF5 trajectory
    target_pose = get_hdf5_target_positions(env, horizon=1)
    if target_pose.ndim == 2 and target_pose.shape[1] > 6:
        target_pose = target_pose[:, :6]
    elif target_pose.ndim == 3:
        target_pose = target_pose.squeeze(1)

    # (3) Reward 계산
    e_pose = ee_pose - target_pose
    w = torch.tensor([1.0, 1.0, 1.0, 0.3, 0.3, 0.3], device=device).unsqueeze(0)
    e_sq = (w * e_pose) ** 2
    k_pos = 3.0
    r_pose_axiswise = torch.exp(-k_pos * e_sq)
    reward = torch.mean(r_pose_axiswise, dim=1)  # (N,)

    # (4) History 기록
    global _position_tracking_history, _position_reward_history
    _position_tracking_history.append((step, target_pose[0].cpu().numpy(), ee_pose[0].cpu().numpy()))
    _position_reward_history.append((step, reward[0].item()))

    # (5) Console Debug Log
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

    # (6) Episode 종료 시 시각화
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
    plt.title(f"Joint Tracking ({version})")
    plt.xlabel("Step"); plt.ylabel("Joint [rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_tracking_{version}_ep{_episode_counter_joint+1}.png"))
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
    plt.title(f"Per-Joint Reward + Total Reward ({version})")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_joint_{version}_ep{_episode_counter_joint+1}.png"))
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
    plt.title(f"EE 6D Pose Tracking ({version})")
    plt.xlabel("Step"); plt.ylabel("Pose [m/rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pos_tracking_{version}_ep{_episode_counter_position+1}.png"))
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
    plt.title(f"6D Pose Reward ({version})")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_pos_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    _position_tracking_history.clear()
    _position_reward_history.clear()
    _episode_counter_position += 1
