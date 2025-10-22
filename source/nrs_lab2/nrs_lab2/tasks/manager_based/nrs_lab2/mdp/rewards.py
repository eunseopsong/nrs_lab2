# SPDX-License-Identifier: BSD-3-Clause
"""
v23.0: Dual Tracking Reward (Joint + Cartesian Position, Unwrapped Visualization)
--------------------------------------------------------------------------------
- Reward Type: Exponential kernel (no tanh)
- Goal: Joint-space + Cartesian-space 병렬 학습 + 안정적 시각화

Changes from v22:
    ✅ roll, pitch, yaw 시각화 시 ±pi wrap 문제 해결 (np.unwrap)
    ✅ reward 계산은 기존과 동일 (±pi 내부에서 error 계산)
    ✅ 코드 주석 정리 및 버전 자동 반영 구조 유지

Joint Reward:
    - q₂ 안정화 + velocity damping + weighted bias correction
    - Target: get_hdf5_target_joints(env, horizon=8)

Position Reward:
    - 6DoF (x, y, z, roll, pitch, yaw)
    - Target: get_hdf5_target_positions(env, horizon=1)
    - FK 기반 pose 사용

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
version = "v23"  # ✅ version 변수로 모든 title 및 파일명 자동 관리

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
# (2) Position Tracking Reward (6DoF, with Velocity Term)
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """6D End-effector pose + velocity tracking reward (FKSolver 기반, exponential kernel)"""
    from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import get_ee_pose

    device = env.device
    step = int(env.common_step_counter)

    # (1) EE pose from FKSolver
    ee_pose = get_ee_pose(env)  # (N, 6): [x, y, z, roll, pitch, yaw]

    # (2) EE velocity (world frame)
    robot = env.scene["robot"]
    link_idx = robot.find_bodies("wrist_3_link")[0]
    lin_vel = robot.data.body_lin_vel_w[:, link_idx, :].squeeze(1)  # (N,3)
    ang_vel = robot.data.body_ang_vel_w[:, link_idx, :].squeeze(1)  # (N,3)
    ee_vel = torch.cat([lin_vel, ang_vel], dim=1)  # (N,6)

    # (3) Target pose from HDF5 trajectory
    target_pose = get_hdf5_target_positions(env, horizon=2)
    if target_pose.ndim == 3:
        target_pose = target_pose.squeeze(1)
    if target_pose.shape[1] > 12:
        target_pose = target_pose[:, :12]

    # horizon=2 → 첫 6개는 현재, 다음 6개는 다음 시점
    target_curr, target_next = target_pose[:, :6], target_pose[:, 6:12]
    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    target_vel = (target_next - target_curr) / (dt + 1e-8)

    # (4) Errors
    e_pose = ee_pose - target_next
    e_vel = ee_vel - target_vel

    # (5) Weights & gains
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device).unsqueeze(0)
    k_pose = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0, 4.0], device=device).unsqueeze(0)
    k_vel = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], device=device).unsqueeze(0)

    e_pose2 = (w * e_pose) ** 2
    e_vel2 = (w * e_vel) ** 2

    r_pose_axiswise = torch.exp(-k_pose * e_pose2)
    r_vel_axiswise = torch.exp(-k_vel * e_vel2)

    r_pose = torch.sum(r_pose_axiswise, dim=1)
    r_vel = torch.sum(r_vel_axiswise, dim=1)
    total = 0.9 * r_pose + 0.1 * r_vel

    # (6) History 기록
    global _position_tracking_history, _position_reward_history
    _position_tracking_history.append((step, target_next[0].cpu().numpy(), ee_pose[0].cpu().numpy()))
    _position_reward_history.append((step, total[0].item()))

    # (7) Console Debug Log
    if step % 10 == 0:
        mean_e_pose = torch.mean(torch.abs(e_pose), dim=0).cpu().numpy()
        mean_e_vel = torch.mean(torch.abs(e_vel), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_axiswise, dim=0).cpu().numpy()
        mean_r_vel = torch.mean(r_vel_axiswise, dim=0).cpu().numpy()
        print(f"[Position Step {step:>5}] |e_pose|={torch.norm(e_pose, dim=1).mean():.4f}, |e_vel|={torch.norm(e_vel, dim=1).mean():.4f}, total={total.mean():.4f}")
        labels = ["x", "y", "z", "roll", "pitch", "yaw"]
        for i in range(6):
            print(
                f"  {labels[i]:<6} | e_pose={mean_e_pose[i]:+.4f} | e_vel={mean_e_vel[i]:+.4f} | "
                f"r_pose={mean_r_pose[i]:.4f} | r_vel={mean_r_vel[i]:.4f}"
            )

    # (8) Episode 종료 시 시각화
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_position(step)

    return total



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
    """Episode 단위 Cartesian-space (6D) 시각화 (unwrap 적용)"""
    global _position_tracking_history, _position_reward_history, _episode_counter_position

    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_lab2/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    steps, targets, currents = zip(*_position_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)

    # ✅ roll, pitch, yaw unwrap for smooth visualization
    targets[:, 3:6] = np.unwrap(targets[:, 3:6], axis=0)
    currents[:, 3:6] = np.unwrap(currents[:, 3:6], axis=0)

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
