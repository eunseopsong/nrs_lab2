# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for UR10e + Spindle imitation
- Joint command error (L2 penalty)
- Joint command error with tanh kernel (shaping reward)
- Debugging: print current vs target joint states
- Visualization: every 30 seconds (episode length), save plot to ~/nrs_lab2/outputs/png/
"""

from __future__ import annotations
import os
import torch
import numpy as np      # ✅ 추가
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

# observations.py 에 있는 유틸 불러오기
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import get_hdf5_target
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import get_hdf5_target_future

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -------------------
# Global buffer
# -------------------
_joint_tracking_history = []   # (step, target, current) 기록용


# -------------------
# Reward functions
# -------------------
def joint_command_error(env: ManagerBasedRLEnv, command_name=None, asset_cfg=None) -> torch.Tensor:
    """Joint L2 tracking error (penalty)"""
    global _joint_tracking_history
    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env).unsqueeze(0).repeat(env.num_envs, 1)
    diff = q - target
    error = torch.norm(diff, dim=-1)

    # 디버깅 출력 (10 step마다)
    if env.common_step_counter % 10 == 0:
        print(f"[Step {env.common_step_counter}]")
        print(f"  Target joints[0]: {target[0].cpu().numpy()}")
        print(f"  Current joints[0]: {q[0].cpu().numpy()}")
        print(f"  L2 Error[0]: {error[0].item():.6f}")

    # 기록 (env 0만 저장)
    _joint_tracking_history.append(
        (env.common_step_counter, target[0].detach().cpu().numpy(), q[0].detach().cpu().numpy())
    )

    # 30초마다 (episode_length_s 기준) 시각화
    if env.common_step_counter > 0 and env.common_step_counter % int(env.max_episode_length) == 0:
        save_joint_tracking_plot(env)

    return -error  # penalty


def joint_command_error_tanh(env: ManagerBasedRLEnv, std: float = 0.1, command_name=None, asset_cfg=None) -> torch.Tensor:
    """Joint L2 tracking error with tanh shaping"""
    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env).unsqueeze(0).repeat(env.num_envs, 1)
    diff = q - target
    distance = torch.norm(diff, dim=-1)
    reward = 1 - torch.tanh(distance / std)

    # 디버깅 출력 (10 step마다)
    if env.common_step_counter % 10 == 0:
        print(f"[Step {env.common_step_counter}]")
        print(f"  Target joints[0]: {target[0].cpu().numpy()}")
        print(f"  Current joints[0]: {q[0].cpu().numpy()}")
        print(f"  tanh Reward[0]: {reward[0].item():.6f}")

    return reward




# -------------------
# Joint tracking reward (v7: position + velocity error + joint-wise clipped debug)
# -------------------

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

_joint_bounds = None
_episode_counter = 0


def joint_tracking_reward(env: "ManagerBasedRLEnv"):
    """
    v7: Position + Velocity reward
    ------------------------------
    r = w_pos * exp(-k_pose * clip(||q - q*||_2 / max_pos_error, 0, 1))
      + w_vel * exp(-k_vel  * clip(||qd - qd*||_2 / max_vel_error, 0, 1))

    - position: 2 rad 기준
    - velocity: 3 rad/s 기준
    """

    # ---------------------------------------------------------
    # (1) Basic setup
    # ---------------------------------------------------------
    robot = env.scene["robot"]
    q = robot.data.joint_pos[:, :6]
    qd = robot.data.joint_vel[:, :6]
    dt = getattr(env.sim, "dt", 1.0 / 120.0) * getattr(env, "decimation", 1)

    # ---------------------------------------------------------
    # (2) Target horizon: single-step (no exponential decay)
    # ---------------------------------------------------------
    fut = get_hdf5_target_future(env, horizon=2)
    D = q.shape[1]
    q_star_next = fut[:, D:2 * D]
    q_star_curr = fut[:, :D]
    qd_star = (q_star_next - q_star_curr) / (dt + 1e-8)

    # ---------------------------------------------------------
    # (3) Position error (linear scaled clipping)
    # ---------------------------------------------------------
    e_q = q - q_star_next
    l2_error_pos = torch.norm(e_q, dim=1)
    max_pos_error = 2.0
    l2_error_pos_clipped = torch.clamp(l2_error_pos / max_pos_error, 0.0, 0.5)
    jointwise_pos_clipped = torch.clamp(torch.abs(e_q) / max_pos_error, 0.0, 0.5)
    r_pos = torch.exp(-8.0 * l2_error_pos_clipped)

    # ---------------------------------------------------------
    # (4) Velocity error (linear scaled clipping)
    # ---------------------------------------------------------
    e_qd = qd - qd_star
    l2_error_vel = torch.norm(e_qd, dim=1)
    max_vel_error = 3.0
    l2_error_vel_clipped = torch.clamp(l2_error_vel / max_vel_error, 0.0, 1.0)
    jointwise_vel_clipped = torch.clamp(torch.abs(e_qd) / max_vel_error, 0.0, 1.0)
    r_vel = torch.exp(-3.0 * l2_error_vel_clipped)

    # ---------------------------------------------------------
    # (5) Weighted total reward
    # ---------------------------------------------------------
    w_pos = 0.8
    w_vel = 0.2
    total = w_pos * r_pos + w_vel * r_vel

    # ---------------------------------------------------------
    # (6) Debug print (every 10 steps)
    # ---------------------------------------------------------
    step = int(env.common_step_counter)
    if step % 10 == 0:
        with torch.no_grad():
            mean_pos_raw = l2_error_pos.mean().item()
            mean_vel_raw = l2_error_vel.mean().item()
            mean_pos_clip = l2_error_pos_clipped.mean().item()
            mean_vel_clip = l2_error_vel_clipped.mean().item()
            r_val = total[0].item()

            print(f"[Step {step}] v7: Position + Velocity reward")
            print(f"  mean |e_q|_2 (raw)   = {mean_pos_raw:.4f}")
            print(f"  mean |e_qd|_2 (raw)  = {mean_vel_raw:.4f}")
            print(f"  mean |e_q|_2 (clip)  = {mean_pos_clip:.4f}")
            print(f"  mean |e_qd|_2 (clip) = {mean_vel_clip:.4f}")
            print(f"  r_pos (env0)         = {r_pos[0].item():.6f}")
            print(f"  r_vel (env0)         = {r_vel[0].item():.6f}")
            print(f"  total (env0)         = {r_val:.6f}")
            print(f"  Target (t+1)[0]:     {q_star_next[0].detach().cpu().numpy()}")
            print(f"  Current joints[0]:   {q[0].detach().cpu().numpy()}")
            print(f"  Error (q - q* )[0]:  {e_q[0].detach().cpu().numpy()}")
            print(f"  Error (qd - qd* )[0]:{e_qd[0].detach().cpu().numpy()}")

            # 🔹 각 조인트별 clipped error 출력
            pos_clip_vals = jointwise_pos_clipped[0].detach().cpu().numpy()
            vel_clip_vals = jointwise_vel_clipped[0].detach().cpu().numpy()
            pos_clip_str = ", ".join([f"{v:.3f}" for v in pos_clip_vals])
            vel_clip_str = ", ".join([f"{v:.3f}" for v in vel_clip_vals])
            print(f"  Joint-wise pos clipped errors: [{pos_clip_str}]")
            print(f"  Joint-wise vel clipped errors: [{vel_clip_str}]")
            print("-" * 90)

    # ---------------------------------------------------------
    # (7) History & Visualization (env0 only)
    # ---------------------------------------------------------
    if "_joint_tracking_history" in globals():
        globals()["_joint_tracking_history"].append(
            (step, q_star_next[0].detach().cpu().numpy(), q[0].detach().cpu().numpy())
        )

    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        if step > 0 and step % int(env.max_episode_length) == 0:
            if "_joint_tracking_history" in globals() and globals()["_joint_tracking_history"]:
                save_joint_tracking_plot(env)

    return total





# --------------------------------
# Reward improvement (meta reward)
# --------------------------------
def reward_convergence_boost(env, current_reward: torch.Tensor, alpha: float = 6.0, sensitivity: float = 0.2):
    """
    안정화된 수렴 보상 (Soft Exponential kernel)
    - 작은 Δreward → 선형 증가
    - 큰 Δreward → 포화(saturate) → 진동 방지
    - alpha: 보상 강도
    - sensitivity: 민감도 (클수록 완화)
    """
    global _prev_total_reward

    if _prev_total_reward is None or torch.isnan(current_reward).any():
        _prev_total_reward = current_reward.clone()
        return torch.zeros_like(current_reward)

    # Reward 변화량
    reward_delta = current_reward - _prev_total_reward

    # Soft exponential kernel (exp 대신 tanh 기반)
    # exp 커널의 과도한 폭발 방지용 -> 부호 유지 + saturation
    reward_boost = alpha * torch.tanh(reward_delta / sensitivity)

    # 업데이트
    _prev_total_reward = current_reward.clone()

    return reward_boost







# -------------------
# Contact Force reward
# -------------------


def contact_force_reward(env: ManagerBasedRLEnv,
                         sensor_name: str = "contact_forces",
                         fz_min: float = 5.0,
                         fz_max: float = 15.0,
                         margin: float = 2.0,
                         weight: float = 1.0) -> torch.Tensor:
    """
    Reward for maintaining end-effector normal force (Fz) within [fz_min, fz_max].

    Args:
        env: Isaac Lab ManagerBasedRLEnv environment.
        sensor_name: Contact sensor name (default: 'contact_forces').
        fz_min, fz_max: desired Fz range [N].
        margin: tolerance margin for smooth tanh decay outside range.
        weight: scaling factor for the reward.

    Returns:
        torch.Tensor(num_envs,) : per-env scalar reward.
    """
    # ✅ Contact sensor 읽기
    sensor = env.scene.sensors[sensor_name]
    forces_w = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)

    # 여러 rigid body가 있으면 평균 Fz 사용
    mean_force = torch.mean(forces_w, dim=1)  # (num_envs, 3)
    fz = mean_force[:, 2]  # Fz (z축 성분)

    # ✅ 보상 계산: Fz가 [fz_min, fz_max] 범위 내이면 +1, 아니면 0~음수
    # Smooth transition using tanh kernel
    lower_smooth = torch.tanh((fz - fz_min) / margin)
    upper_smooth = torch.tanh((fz_max - fz) / margin)
    reward_raw = 0.5 * (lower_smooth + upper_smooth)

    # scale and clip
    reward = weight * torch.clamp(reward_raw, 0.0, 1.0)

    # ✅ 디버깅 출력 (100 step마다)
    if env.common_step_counter % 1000 == 0:
        print(f"[ContactReward DEBUG] Step {env.common_step_counter}: Fz={fz[0].item():.3f}, Reward={reward[0].item():.3f}")

    return reward

# -----------------------------------------------------------------------------
# Camera Distance Reward
# -----------------------------------------------------------------------------

def camera_distance_reward(env, target_distance: float = 0.185, sigma: float = 0.02):
    """
    Reward for maintaining camera distance near the spindle length (≈0.185m).

    Args:
        env: ManagerBasedRLEnv
        target_distance (float): desired mean camera distance (m)
        sigma (float): Gaussian width (how sharply reward decreases)
    """
    # camera sensor 데이터 접근
    camera_data = env.scene["camera"].data.output["distance_to_image_plane"]

    # [num_envs, H, W] → 평균 거리로 축소
    d_mean = torch.mean(camera_data.view(env.num_envs, -1), dim=1)

    # 거리 오차
    error = torch.abs(d_mean - target_distance)

    # Gaussian kernel 기반 보상
    reward = torch.exp(- (error ** 2) / (2 * sigma ** 2))

    # 디버깅 출력 (매 100 step마다)
    if env.common_step_counter % 100 == 0:
        print(f"[CameraReward DEBUG] Step {env.common_step_counter}: mean_dist={d_mean[0].item():.4f} m, "
              f"target={target_distance:.3f}, reward={reward[0].item():.4f}")

        # 결과 저장 (선택)
        save_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"camera_distance_step{env.common_step_counter}.txt")
        with open(save_path, "w") as f:
            f.write(f"Step {env.common_step_counter}, mean_dist={d_mean[0].item():.6f}, reward={reward[0].item():.6f}\n")

    return reward


# -------------------
# Visualization (numpy-only, same API)
# -------------------
_episode_counter = 0   # 전역 카운터 유지

def save_joint_tracking_plot(env: "ManagerBasedRLEnv"):
    """joint target vs current trajectory 시각화 (q1~q6, 같은 번호는 같은 색)"""
    global _joint_tracking_history, _episode_counter

    if not _joint_tracking_history:
        return

    steps, targets, currents = zip(*_joint_tracking_history)
    steps = np.asarray(steps)
    targets = np.asarray(targets)   # [T,6], np.float
    currents = np.asarray(currents) # [T,6]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # tab10 팔레트 (10개 색상)

    for j in range(targets.shape[1]):
        color = colors[j % len(colors)]
        plt.plot(steps, targets[:, j], "--", label=f"Target q{j+1}", color=color, linewidth=1.2)
        plt.plot(steps, currents[:, j], "-",  label=f"Current q{j+1}", color=color, linewidth=2.0)

    plt.xlabel("Step")
    plt.ylabel("Joint Value (rad)")
    plt.title("Joint Tracking (Target vs Current)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)

    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"joint_tracking_episode{_episode_counter}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved joint tracking plot to {save_path}")

    _episode_counter += 1
    _joint_tracking_history = []  # 다음 episode 준비



# -------------------
# Termination function
# -------------------
def reached_end(env: ManagerBasedRLEnv, command_name=None, asset_cfg=None) -> torch.Tensor:
    """Trajectory 끝에 도달하면 종료"""
    from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import _hdf5_trajectory
    if _hdf5_trajectory is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_trajectory.shape[0]
    return torch.tensor(env.common_step_counter >= (T - 1),
                        dtype=torch.bool, device=env.device).repeat(env.num_envs)
