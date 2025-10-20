# SPDX-License-Identifier: BSD-3-Clause
"""
v20.1: q2 Drift-Damped Reward (exp kernel + total reward visualization)
-----------------------------------------------------------------------
- **Reward Type:** Exponential kernel (no tanh)
- **Goal:** 강화된 q2 안정화 및 velocity damping 적용
- **Bias Correction:** Weighted bias 적용
- **Joint Weights (wj):** [1.0, 2.0, 1.0, 4.0, 1.0, 1.0]
- **k_pos:** [1.0, 8.0, 2.0, 6.0, 2.0, 2.0]
- **k_vel:** [0.10, 0.40, 0.10, 0.40, 0.10, 0.10]
- **Home Pose (UR10E_HOME_DICT):**
    shoulder_pan_joint :  0.673993
    shoulder_lift_joint : -1.266343
    elbow_joint         : -2.472206
    wrist_1_joint       : -1.160399
    wrist_2_joint       :  1.479353
    wrist_3_joint       :  1.324695
- 각 episode 종료 시:
    (1) Joint Tracking Plot 저장 → ~/nrs_lab2/outputs/png/
    (2) Reward per Joint + Total Reward Plot 저장 → ~/nrs_lab2/outputs/rewards/
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import torch, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import get_hdf5_target_future

# -----------------------------------------------------------
# Global States
# -----------------------------------------------------------
_joint_tracking_history = []
_reward_history = []
_episode_counter = 0


# -----------------------------------------------------------
# Joint Tracking Reward (v20.1)
# -----------------------------------------------------------
def joint_tracking_reward(env: "ManagerBasedRLEnv"):

    robot = env.scene["robot"]
    q  = robot.data.joint_pos[:, :6]
    qd = robot.data.joint_vel[:, :6]
    dt = getattr(env.sim, "dt", 1.0 / 30.0) * getattr(env, "decimation", 1)
    D  = q.shape[1]
    step = int(env.common_step_counter)

    # ✅ 여기 수정
    fut = get_hdf5_target_future(env, horizon=8)
    q_star_curr = fut[:, :D]
    q_star_next = fut[:, D:2*D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    # 이하 동일 (reward 계산 및 시각화 등)


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
        print(f"[Step {step}] mean(e_q)={torch.norm(e_q, dim=1).mean():.3f}, total={total.mean():.3f}")
        mean_e_q_abs = torch.mean(torch.abs(e_q), dim=0).cpu().numpy()
        mean_r_pose = torch.mean(r_pose_jointwise, dim=0).cpu().numpy()
        for j in range(D):
            print(f"  joint{j+1}: |mean(e_q)|={mean_e_q_abs[j]:.3f}, r_pose={mean_r_pose[j]:.3f}")

    # -------- history 저장 --------
    _joint_tracking_history.append((step, q_star_next[0].cpu().numpy(), q[0].cpu().numpy()))
    _reward_history.append((step, r_pose_jointwise[0].cpu().numpy()))

    # -------- episode 종료 시 시각화 --------
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        episode_steps = int(env.max_episode_length)
        if step > 0 and (step % episode_steps == episode_steps - 1):
            save_episode_plots(step)

    return total


# -----------------------------------------------------------
# Visualization (Joint Tracking + Reward)
# -----------------------------------------------------------
def save_episode_plots(step: int):
    """각 episode 종료 시 joint tracking 및 reward 시각화"""
    global _joint_tracking_history, _reward_history, _episode_counter

    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png/")
    reward_dir = os.path.expanduser("~/nrs_lab2/outputs/rewards/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    # -------------------------------
    # (1) Joint Tracking Plot
    # -------------------------------
    steps, targets, currents = zip(*_joint_tracking_history)
    targets, currents = np.vstack(targets), np.vstack(currents)
    colors = ["r","g","b","orange","purple","gray"]

    plt.figure(figsize=(10,6))
    for j in range(targets.shape[1]):
        plt.plot(targets[:,j],"--",color=colors[j],label=f"Target q{j+1}")
        plt.plot(currents[:,j],"-",color=colors[j],label=f"Current q{j+1}")
    plt.legend(); plt.grid(True)
    plt.title(f"Joint Tracking (v20.1)")
    plt.xlabel("Step"); plt.ylabel("Joint [rad]")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"joint_tracking_v20_ep{_episode_counter+1}.png"))
    plt.close()

    # -------------------------------
    # (2) Reward Visualization (per joint + total)
    # -------------------------------
    def smooth(y, window=50):
        if len(y) < window:
            return y
        cumsum = np.cumsum(np.insert(y, 0, 0))
        return (cumsum[window:] - cumsum[:-window]) / window

    r_steps, r_values = zip(*_reward_history)
    r_values = np.vstack(r_values)                # shape: [T, 6]
    total_reward = np.sum(r_values, axis=1)       # shape: [T,]
    smooth_window = 50

    plt.figure(figsize=(10,6))
    for j in range(r_values.shape[1]):
        smooth_y = smooth(r_values[:, j], smooth_window)
        smooth_x = np.linspace(r_steps[0], r_steps[-1], len(smooth_y))
        plt.plot(smooth_x, smooth_y, color=colors[j], label=f"r_pose(q{j+1})")

    # ----- total reward (black thick line) -----
    smooth_total = smooth(total_reward, smooth_window)
    smooth_x_total = np.linspace(r_steps[0], r_steps[-1], len(smooth_total))
    plt.plot(smooth_x_total, smooth_total, color="black", linewidth=2.5, label="Total Reward")

    plt.legend(); plt.grid(True)
    plt.title("Per-Joint Pose Reward + Total Reward (Smoothed, v20.1)")
    plt.xlabel("Step"); plt.ylabel("Reward value")
    plt.tight_layout()
    plt.savefig(os.path.join(reward_dir, f"r_pose_total_v20_ep{_episode_counter+1}.png"))
    plt.close()

    # -------------------------------
    # (3) Cleanup
    # -------------------------------
    _joint_tracking_history.clear()
    _reward_history.clear()
    _episode_counter += 1


# -----------------------------------------------------------
# Contact Force Reward
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
# Camera Distance Reward
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
# Termination Condition
# -----------------------------------------------------------
def reached_end(env: ManagerBasedRLEnv, command_name=None, asset_cfg=None) -> torch.Tensor:
    from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import _hdf5_trajectory
    if _hdf5_trajectory is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_trajectory.shape[0]
    return torch.tensor(env.common_step_counter >= (T - 1),
                        dtype=torch.bool, device=env.device).repeat(env.num_envs)
