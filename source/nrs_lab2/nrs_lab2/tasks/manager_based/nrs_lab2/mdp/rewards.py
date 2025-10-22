# SPDX-License-Identifier: BSD-3-Clause
"""
v25.0: Dual Tracking Reward (Joint + Cartesian, Wrap-Safe Orientation)
-----------------------------------------------------------------------
- Reward Type: Exponential kernel (no tanh)
- Goal: Joint-space + Cartesian-space 병렬 학습 + Orientation wrap-safe error 계산

Changes from v24:
    ✅ Orientation (roll/pitch/yaw) error 계산 시 wrap-around 보정 추가
       → angle_diff() 함수를 통해 ±π 경계에서 연속적 오차 계산
    ✅ FK 기반 EE pose/velocity tracking 유지
    ✅ 시각화(np.unwrap) 유지
    ✅ version 변수 자동 반영 구조 유지

Joint Reward:
    - q₂ 안정화 + velocity damping + weighted bias correction
    - Target: get_hdf5_target_joints(env, horizon=8)
    - Output: total = 0.9 * r_pose + 0.1 * r_vel

Position Reward:
    - 6DoF (x, y, z, roll, pitch, yaw) + 6D velocity (vx, vy, vz, wx, wy, wz)
    - Target: get_hdf5_target_positions(env, horizon=2)
    - Orientation error wrap-safe 계산
    - Output: total = 0.9 * r_pose + 0.1 * r_vel

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
    get_ee_pose,
)

# -----------------------------------------------------------
# Global
# -----------------------------------------------------------
version = "v25"
_joint_tracking_history = []
_joint_reward_history = []
_position_tracking_history = []
_position_reward_history = []
_episode_counter_joint = 0
_episode_counter_position = 0


# -----------------------------------------------------------
# Utility: angle wrap correction
# -----------------------------------------------------------
def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute minimal difference between two angles (in radians), wrapped to [-pi, pi]."""
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff


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

    fut = get_hdf5_target_joints(env, horizon=8)
    q_star_curr, q_star_next = fut[:, :D], fut[:, D:2*D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    e_q, e_qd = q - q_star_next, qd - qd_star

    wj = torch.tensor([1, 2.0, 1, 4.0, 1, 1], device=q.device).unsqueeze(0)
    k_pos = torch.tensor([1.0, 8.0, 2.0, 6.0, 2.0, 2.0], device=q.device).unsqueeze(0)
    k_vel = torch.tensor([0.10, 0.40, 0.10, 0.40, 0.10, 0.10], device=q.device).unsqueeze(0)

    e_q2, e_qd2 = wj * (e_q ** 2), wj * (e_qd ** 2)
    r_pose_jointwise = torch.exp(-k_pos * e_q2)
    r_vel_jointwise = torch.exp(-k_vel * e_qd2)

    r_pose, r_vel = r_pose_jointwise.sum(dim=1), r_vel_jointwise.sum(dim=1)
    total = 0.9 * r_pose + 0.1 * r_vel

    if step % 10 == 0:
        print(f"[Joint Step {step}] mean(e_q)={torch.norm(e_q, dim=1).mean():.3f}, total={total.mean():.3f}")
        mean_e_q_abs = torch.mean(torch.abs(e_q), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_jointwise, dim=0).cpu().numpy()
        for j in range(D):
            print(f"  joint{j+1}: |mean(e_q)|={mean_e_q_abs[j]:.3f}, r_pose={mean_r_pose[j]:.3f}")

    _joint_tracking_history.append((step, q_star_next[0].cpu().numpy(), q[0].cpu().numpy()))
    _joint_reward_history.append((step, r_pose_jointwise[0].cpu().numpy()))

    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_joint(step)

    return total


# -----------------------------------------------------------
# (2) Position Tracking Reward (6D + velocity)
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """6D EE pose + velocity tracking reward (wrap-safe orientation)"""
    device = env.device
    step = int(env.common_step_counter)

    # (1) FK 기반 EE pose
    ee_pose = get_ee_pose(env)  # (N,6): [x,y,z,roll,pitch,yaw]
    robot = env.scene["robot"]
    ee_vel = robot.data.body_lin_vel_w[:, robot.find_bodies("wrist_3_link")[0], :].squeeze(1)
    ee_ang = robot.data.body_ang_vel_w[:, robot.find_bodies("wrist_3_link")[0], :].squeeze(1)
    ee_vel6d = torch.cat([ee_vel, ee_ang], dim=1)

    # (2) HDF5 target (2-step horizon)
    fut = get_hdf5_target_positions(env, horizon=2)
    if fut.ndim == 3: fut = fut.squeeze(1)
    target_curr, target_next = fut[:, :6], fut[:, 6:12]
    target_vel = (target_next - target_curr) / (1.0 / 30.0)

    # (3) wrap-safe orientation difference
    e_pose = ee_pose.clone()
    e_pose[:, :3] -= target_next[:, :3]
    diff_ori = angle_diff(ee_pose[:, 3:6].cpu().numpy(), target_next[:, 3:6].cpu().numpy())
    e_pose[:, 3:6] = torch.from_numpy(diff_ori).to(device)

    # (4) velocity error
    e_vel = ee_vel6d - target_vel

    # (5) reward
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device).unsqueeze(0)
    k_pose, k_vel = 8.0, 0.2
    r_pose_axiswise = torch.exp(-k_pose * (w * e_pose) ** 2)
    r_vel_axiswise  = torch.exp(-k_vel  * (w * e_vel) ** 2)
    r_pose, r_vel = torch.mean(r_pose_axiswise, dim=1), torch.mean(r_vel_axiswise, dim=1)
    reward = 0.9 * r_pose + 0.1 * r_vel

    # (6) 기록 및 로그
    global _position_tracking_history, _position_reward_history
    _position_tracking_history.append((step, target_next[0].cpu().numpy(), ee_pose[0].cpu().numpy()))
    _position_reward_history.append((step, reward[0].item()))

    if step % 10 == 0:
        mean_e_pose = torch.mean(torch.abs(e_pose), dim=0).cpu().numpy()
        mean_e_vel  = torch.mean(torch.abs(e_vel), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_axiswise, dim=0).cpu().numpy()
        mean_r_vel  = torch.mean(r_vel_axiswise, dim=0).cpu().numpy()
        print(f"[Position Step {step}] |e_pose|={torch.norm(e_pose,dim=1).mean():.4f}, |e_vel|={torch.norm(e_vel,dim=1).mean():.4f}, total={reward.mean():.4f}")
        labels = ["x","y","z","roll","pitch","yaw"]
        for i in range(6):
            print(f"  {labels[i]:<6} | e_pose={mean_e_pose[i]:+6.4f} | e_vel={mean_e_vel[i]:+6.4f} | r_pose={mean_r_pose[i]:.4f} | r_vel={mean_r_vel[i]:.4f}")

    # (7) 시각화
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_position(step)

    return reward


# -----------------------------------------------------------
# Visualization (공통)
# -----------------------------------------------------------
def save_episode_plots_joint(step: int):
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
    plt.figure(figsize=(10,5))
    plt.plot(r_steps, total_reward, "k", linewidth=2.0, label="Total Reward")
    plt.legend(); plt.grid(True)
    plt.title(f"Joint Reward ({version})")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_joint_{version}_ep{_episode_counter_joint+1}.png"))
    plt.close()

    _joint_tracking_history.clear()
    _joint_reward_history.clear()
    _episode_counter_joint += 1


def save_episode_plots_position(step: int):
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
    plt.figure(figsize=(10,5))
    plt.plot(r_steps, r_values, "g", linewidth=2.5, label="r_pose(6D pose)")
    plt.legend(); plt.grid(True)
    plt.title(f"6D Pose Reward ({version})")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_pos_{version}_ep{_episode_counter_position+1}.png"))
    plt.close()

    _position_tracking_history.clear()
    _position_reward_history.clear()
    _episode_counter_position += 1
