# SPDX-License-Identifier: BSD-3-Clause
"""
v21.0: Dual Tracking Reward (Joint + Cartesian Position)
---------------------------------------------------------
- **Reward Type:** Exponential kernel (no tanh)
- **Goal:** Joint-space + Cartesian-space 병렬 학습 지원  
- **Joint Reward (q-space):** q₂ 안정화 + velocity damping + weighted bias correction  
- **Position Reward (x-space):** End-effector 위치 기반 exponential decay  
- **Visualization:** Episode 종료 시 joint/position 각각 별도 그래프 저장

- **Joint Parameters**
    - **Joint Weights (wj):** [1.0, 2.0, 1.0, 4.0, 1.0, 1.0]
    - **k_pos:** [1.0, 8.0, 2.0, 6.0, 2.0, 2.0]
    - **k_vel:** [0.10, 0.40, 0.10, 0.40, 0.10, 0.10]
    - **Target Source:** get_hdf5_target_joints(env, horizon=8)

- **Position Parameters**
    - **k_pos:** 6.0
    - **Target Source:** get_hdf5_target_positions(env, horizon=8)
    - **Error Metric:** Cartesian distance (‖p - p*‖²)

- **Home Pose (UR10E_HOME_DICT):**
    shoulder_pan_joint :  0.673993
    shoulder_lift_joint : -1.266343
    elbow_joint         : -2.472206
    wrist_1_joint       : -1.160399
    wrist_2_joint       :  1.479353
    wrist_3_joint       :  1.324695

- **Episode End Visualization**
    (1) Joint Tracking → ~/nrs_lab2/outputs/png/joint_tracking_v21_epX.png  
    (2) Position Tracking → ~/nrs_lab2/outputs/png/pos_tracking_v21_epX.png  
    (3) Joint Reward Summary → ~/nrs_lab2/outputs/rewards/r_pose_total_joint_v21_epX.png  
    (4) Position Reward Summary → ~/nrs_lab2/outputs/rewards/r_pose_total_pos_v21_epX.png
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
    """Joint-space tracking reward with exponential kernel (v20.1 style)"""
    robot = env.scene["robot"]
    q  = robot.data.joint_pos[:, :6]
    qd = robot.data.joint_vel[:, :6]
    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    D  = q.shape[1]
    step = int(env.common_step_counter)

    # ✅ Horizon-based target
    fut = get_hdf5_target_joints(env, horizon=8)
    q_star_curr = fut[:, :D]
    q_star_next = fut[:, D:2*D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    # -------- errors --------
    e_q, e_qd = q - q_star_next, qd - qd_star

    # -------- weights & gains --------
    wj = torch.tensor([1, 2.0, 1, 4.0, 1, 1], device=q.device).unsqueeze(0)
    k_pos = torch.tensor([1.0, 8.0, 2.0, 6.0, 2.0, 2.0], device=q.device).unsqueeze(0)
    k_vel = torch.tensor([0.10, 0.40, 0.10, 0.40, 0.10, 0.10], device=q.device).unsqueeze(0)

    e_q2, e_qd2 = wj * (e_q ** 2), wj * (e_qd ** 2)

    # -------- rewards --------
    r_pose_jointwise = torch.exp(-k_pos * e_q2)
    r_vel_jointwise  = torch.exp(-k_vel * e_qd2)
    r_pose, r_vel = r_pose_jointwise.sum(dim=1), r_vel_jointwise.sum(dim=1)
    total = 0.9 * r_pose + 0.1 * r_vel

    # -------- console log every 10 steps --------
    if step % 10 == 0:
        print(f"[Joint Step {step}] mean(e_q)={torch.norm(e_q, dim=1).mean():.3f}, total={total.mean():.3f}")
        mean_e_q_abs = torch.mean(torch.abs(e_q), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_jointwise, dim=0).cpu().numpy()
        for j in range(D):
            print(f"  joint{j+1}: |mean(e_q)|={mean_e_q_abs[j]:.3f}, r_pose={mean_r_pose[j]:.3f}")

    # -------- history 저장 --------
    _joint_tracking_history.append((step, q_star_next[0].cpu().numpy(), q[0].cpu().numpy()))
    _joint_reward_history.append((step, r_pose_jointwise[0].cpu().numpy()))

    # -------- episode 종료 시 시각화 --------
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots_joint(step)

    return total


# -----------------------------------------------------------
# (2) Position Tracking Reward
# -----------------------------------------------------------
def position_tracking_reward(env: "ManagerBasedRLEnv"):
    """Cartesian position tracking reward (exp kernel + velocity term, horizon-based)"""
    global _position_tracking_history, _position_reward_history

    robot = env.scene["robot"]
    ee_pos = robot.data.ee_pos_w[:, 0, :]             # (num_envs, 3)
    ee_vel = robot.data.ee_lin_vel_w[:, 0, :]         # (num_envs, 3)
    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    step = int(env.common_step_counter)

    # ✅ Horizon-based position target (horizon = 8)
    horizon = 8
    fut = get_hdf5_target_positions(env, horizon=horizon)
    D = fut.shape[1] // horizon  # 3
    p_star_curr = fut[:, :D]
    p_star_next = fut[:, D:2*D]
    v_star = (p_star_next - p_star_curr) / (dt + 1e-8)

    # -------- errors --------
    e_p = ee_pos - p_star_next
    e_v = ee_vel - v_star

    # -------- gains --------
    k_pos = torch.tensor([6.0], device=ee_pos.device)
    k_vel = torch.tensor([0.5], device=ee_pos.device)

    # -------- exponential kernel rewards --------
    e_p2 = torch.sum(e_p ** 2, dim=1)
    e_v2 = torch.sum(e_v ** 2, dim=1)

    r_pose = torch.exp(-k_pos * e_p2)
    r_vel  = torch.exp(-k_vel * e_v2)

    # -------- total reward --------
    total = 0.7 * r_pose + 0.3 * r_vel

    # -------- debug print (10-step마다) --------
    if step % 10 == 0:
        print(
            f"[Position Step {step}] mean(|e_p|)={torch.norm(e_p, dim=1).mean():.3f}, "
            f"mean(|e_v|)={torch.norm(e_v, dim=1).mean():.3f}, "
            f"r_pose={r_pose.mean():.3f}, r_vel={r_vel.mean():.3f}, total={total.mean():.3f}"
        )

    # -------- history 저장 --------
    _position_tracking_history.append((step, p_star_next[0].cpu().numpy(), ee_pos[0].cpu().numpy()))
    _position_reward_history.append((step, r_pose[0].cpu().numpy(), r_vel[0].cpu().numpy(), total[0].cpu().numpy()))

    # -------- episode 종료 시 시각화 --------
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

    # -------------------------------
    # (1) Joint Tracking Plot
    # -------------------------------
    steps, targets, currents = zip(*_joint_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)
    colors = ["r", "g", "b", "orange", "purple", "gray"]

    plt.figure(figsize=(10, 6))
    for j in range(targets.shape[1]):
        plt.plot(targets[:, j], "--", color=colors[j], label=f"Target q{j+1}")
        plt.plot(currents[:, j], "-", color=colors[j], label=f"Current q{j+1}")
    plt.legend(); plt.grid(True)
    plt.title("Joint Tracking (v20.1)")
    plt.xlabel("Step"); plt.ylabel("Joint [rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_tracking_v20_ep{_episode_counter_joint+1}.png"))
    plt.close()

    # -------------------------------
    # (2) Reward Visualization
    # -------------------------------
    r_steps, r_values = zip(*_joint_reward_history)
    r_values = np.vstack(r_values)
    total_reward = np.sum(r_values, axis=1)

    def smooth(y, window=50):
        if len(y) < window:
            return y
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    plt.figure(figsize=(10, 6))
    for j in range(r_values.shape[1]):
        smooth_y = smooth(r_values[:, j])
        smooth_x = np.linspace(r_steps[0], r_steps[-1], len(smooth_y))
        plt.plot(smooth_x, smooth_y, color=colors[j], label=f"r_pose(q{j+1})")

    smooth_total = smooth(total_reward)
    smooth_x_total = np.linspace(r_steps[0], r_steps[-1], len(smooth_total))
    plt.plot(smooth_x_total, smooth_total, color="black", linewidth=2.5, label="Total Reward")

    plt.legend(); plt.grid(True)
    plt.title("Per-Joint Reward + Total Reward (v20.1)")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_joint_v20_ep{_episode_counter_joint+1}.png"))
    plt.close()

    _joint_tracking_history.clear()
    _joint_reward_history.clear()
    _episode_counter_joint += 1


# -----------------------------------------------------------
# (4) Visualization: Position Tracking
# -----------------------------------------------------------
def save_episode_plots_position(step: int):
    """Episode 단위 Cartesian-space 시각화"""
    global _position_tracking_history, _position_reward_history, _episode_counter_position

    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_lab2/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    # (1) Position Tracking
    steps, targets, currents = zip(*_position_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)
    colors = ["r", "g", "b"]

    plt.figure(figsize=(10, 6))
    for j in range(3):
        plt.plot(targets[:, j], "--", color=colors[j], label=f"Target pos[{j}]")
        plt.plot(currents[:, j], "-", color=colors[j], label=f"Current pos[{j}]")
    plt.legend(); plt.grid(True)
    plt.title("Position Tracking (v20.1)")
    plt.xlabel("Step"); plt.ylabel("Position [m]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pos_tracking_v20_ep{_episode_counter_position+1}.png"))
    plt.close()

    # (2) Reward Visualization
    r_steps, r_values = zip(*_position_reward_history)
    r_values = np.array(r_values).flatten()
    def smooth(y, window=50):
        if len(y) < window:
            return y
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    smooth_y = smooth(r_values)
    smooth_x = np.linspace(r_steps[0], r_steps[-1], len(smooth_y))
    plt.figure(figsize=(10, 5))
    plt.plot(smooth_x, smooth_y, "g", linewidth=2.5, label="r_pose(position)")
    plt.legend(); plt.grid(True)
    plt.title("Position Reward (v20.1)")
    plt.xlabel("Step"); plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_pos_v20_ep{_episode_counter_position+1}.png"))
    plt.close()

    _position_tracking_history.clear()
    _position_reward_history.clear()
    _episode_counter_position += 1


# -----------------------------------------------------------
# (5) Contact Force Reward (unchanged)
# -----------------------------------------------------------
def contact_force_reward(env: ManagerBasedRLEnv,
                         sensor_name="contact_forces",
                         fz_min=5.0, fz_max=15.0,
                         margin=2.0, weight=1.0):
    sensor = env.scene.sensors[sensor_name]
    forces_w = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    fz = torch.mean(forces_w, dim=1)[:, 2]  # 평균 Fz

    lower_smooth = torch.tanh((fz - fz_min) / margin)
    upper_smooth = torch.tanh((fz_max - fz) / margin)
    reward = weight * torch.clamp(0.5 * (lower_smooth + upper_smooth), 0.0, 1.0)

    if env.common_step_counter % 1000 == 0:
        print(f"[ContactReward] Step {env.common_step_counter}: Fz={fz[0]:.3f}, Reward={reward[0]:.3f}")
    return reward


# -----------------------------------------------------------
# (6) Camera Distance Reward (unchanged)
# -----------------------------------------------------------
def camera_distance_reward(env, target_distance=0.185, sigma=0.02):
    camera_data = env.scene["camera"].data.output["distance_to_image_plane"]
    d_mean = torch.mean(camera_data.view(env.num_envs, -1), dim=1)
    error = torch.abs(d_mean - target_distance)
    reward = torch.exp(- (error ** 2) / (2 * sigma ** 2))

    if env.common_step_counter % 100 == 0:
        print(f"[CameraReward] Step {env.common_step_counter}: mean={d_mean[0]:.4f}, reward={reward[0]:.4f}")
        save_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"camera_distance_step{env.common_step_counter}.txt"), "w") as f:
            f.write(f"Step {env.common_step_counter}, mean_dist={d_mean[0]:.6f}, reward={reward[0]:.6f}\n")

    return reward


# -----------------------------------------------------------
# (7) Termination Condition (unchanged)
# -----------------------------------------------------------
def reached_end(env: ManagerBasedRLEnv, command_name=None, asset_cfg=None) -> torch.Tensor:
    from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import _hdf5_joints
    if _hdf5_joints is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_joints.shape[0]
    return torch.tensor(env.common_step_counter >= (T - 1),
                        dtype=torch.bool, device=env.device).repeat(env.num_envs)
