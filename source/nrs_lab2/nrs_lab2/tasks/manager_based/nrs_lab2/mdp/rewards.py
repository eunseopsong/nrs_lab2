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
# Joint tracking reward (DeepMimic-style: pose + velocity)
# + action / action-delta regularizers
# + (optional) soft boundary additive penalty
# + episode-end plotting (structure 유지)
# -------------------

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

_prev_total_reward = None
_joint_bounds = None        # percentile bounds fallback
_episode_counter = 0        # for plotting index


def joint_tracking_reward(
    env: "ManagerBasedRLEnv",
    # ----- weights -----
    w_pose: float = 0.6,
    w_vel: float = 0.3,
    w_ee: float = 0.0,          # EE 항을 쓰고 싶으면 >0 로, 기본은 0
    # ----- gaussian kernel scales -----
    k_pose: float = 24.0,
    k_vel: float = 8.0,
    k_ee: float = 4.0,
    # ----- regularizers -----
    lam_u: float = 3e-3,
    lam_du: float = 8e-3,
    # ----- boundary penalty (additive, small) -----
    use_boundary: bool = True,
    k_boundary: float = 3.0,
    margin: float = 0.10,
    gamma_boundary: float = 0.2,   # 최종 보상에서 뺄 가중치(작게)
    bounds_mode: str = "percentile",
):
    """
    DeepMimic 스타일의 모사 리워드:
      r = w_pose * exp(-k_pose * ||q - q*||^2)
        + w_vel  * exp(-k_vel  * ||qd - qd*||^2)
        + w_ee   * exp(-k_ee   * ||ee - ee*||^2)
        - lam_u * |u|_1
        - lam_du * |Δu|_1
        - gamma_boundary * boundary_penalty   (선택적, 가산)

    시각화/히스토리/에피소드-끝 플로팅 등 기존 구조는 유지.
    """

    # ---------------------------------------------------------
    # (0) 상태
    # ---------------------------------------------------------
    robot = env.scene["robot"]
    q   = robot.data.joint_pos[:, :6]   # [N,6]
    qd  = robot.data.joint_vel[:, :6]   # [N,6]

    # 시뮬 타임스텝 (다음 타깃으로부터 qd* 계산에 사용)
    dt = getattr(env.sim, "dt", 1.0/120.0) * getattr(env, "decimation", 1)

    # ---------------------------------------------------------
    # (1) 타깃 (HDF5 → 현재/다음 타깃)
    # ---------------------------------------------------------
    fut = get_hdf5_target_future(env, horizon=2)  # [N, 12] (q*_t, q*_{t+1})
    D = q.shape[1]
    q_star    = fut[:, :D]          # 현재 타깃
    q_star_n  = fut[:, D:2*D]       # 다음 타깃
    qd_star   = (q_star_n - q_star) / (dt + 1e-8)

    # 에러
    e_q  = q  - q_star
    e_qd = qd - qd_star

    # ---------------------------------------------------------
    # (2) DeepMimic-style kernels
    # ---------------------------------------------------------
    # 자세/속도 지수 커널 (합리적 스케일은 k_* 로 조절)
    r_pose = torch.exp(-k_pose * (e_q**2).sum(dim=1))     # [N]
    r_vel  = torch.exp(-k_vel  * (e_qd**2).sum(dim=1))    # [N]

    # (선택) EE 모사: 네가 EE 타깃을 제공할 때만 사용
    r_ee = 0.0
    if w_ee > 0.0 and hasattr(robot.data, "ee_pos_w") and callable(globals().get("get_demo_ee_target", None)):
        ee      = robot.data.ee_pos_w[:, :3]          # [N,3]
        ee_star = get_demo_ee_target(env)             # [N,3]
        r_ee = torch.exp(-k_ee * ((ee - ee_star)**2).sum(dim=1))
    else:
        r_ee = torch.zeros(q.shape[0], device=q.device)

    # ---------------------------------------------------------
    # (3) 액션 규제 (크기/변화율)
    # ---------------------------------------------------------
    u = getattr(env, "_last_action", None)
    if u is None:
        u = torch.zeros_like(q)
    if not hasattr(env, "_prev_u_for_rew"):
        env._prev_u_for_rew = torch.zeros_like(u)

    du = u - env._prev_u_for_rew
    env._prev_u_for_rew = u.detach()

    r_u  = -lam_u  * (u.abs().mean(dim=1))
    r_du = -lam_du * (du.abs().mean(dim=1))

    # ---------------------------------------------------------
    # (4) (선택) 조인트 경계 페널티 : 가산형(부드럽게)
    # ---------------------------------------------------------
    boundary_penalty = torch.zeros(q.shape[0], device=q.device)
    if use_boundary:
        joint_min = joint_max = None
        jl = getattr(robot.data, "joint_pos_limits", None)
        if jl is not None and jl.shape[-1] >= 6:
            # IsaacLab의 [2, dof] 또는 [dof,2] 형태 모두 대비
            if jl.ndim == 2:
                joint_min = jl[0, :6].to(q.device)
                joint_max = jl[1, :6].to(q.device)
            else:
                joint_min = jl[:, 0][:6].to(q.device)
                joint_max = jl[:, 1][:6].to(q.device)
        else:
            # 퍼센타일 기반 fallback (1~99%)
            global _joint_bounds
            if _joint_bounds is None or bounds_mode == "percentile":
                import h5py
                path = "/home/eunseop/nrs_lab2/datasets/joint_recording.h5"
                with h5py.File(path, "r") as f:
                    joint_data = f["joint_positions"][:]   # [T,>=6]
                lo = np.percentile(joint_data[:, :6], 1, axis=0)
                hi = np.percentile(joint_data[:, :6], 99, axis=0)
                _joint_bounds = (
                    torch.tensor(lo, device=q.device, dtype=torch.float32),
                    torch.tensor(hi, device=q.device, dtype=torch.float32),
                )
            joint_min, joint_max = _joint_bounds

        below = torch.clamp((joint_min + margin) - q, min=0.0)
        above = torch.clamp(q - (joint_max - margin), min=0.0)
        overflow = below + above                         # [N,6]
        overflow_mean = overflow.abs().mean(dim=1)       # [N]
        boundary_penalty = 1.0 - torch.exp(-k_boundary * overflow_mean)  # 0~1

    # ---------------------------------------------------------
    # (5) 합성 (가산형)
    # ---------------------------------------------------------
    total = w_pose * r_pose + w_vel * r_vel + w_ee * r_ee + r_u + r_du
    if use_boundary:
        total = total - gamma_boundary * boundary_penalty

    # 아주 작은 하한만 (신호 소실 방지) — 강한 클램프 금지
    total = torch.clamp(total, min=1e-6)

    # ---------------------------------------------------------
    # (6) 디버깅 출력 (10 step마다)
    # ---------------------------------------------------------
    step = int(env.common_step_counter)
    if step % 10 == 0:
        with torch.no_grad():
            print(f"[Step {step}] DeepMimic reward")
            print(f"  k_pose={k_pose}, k_vel={k_vel}, w_pose={w_pose}, w_vel={w_vel}, w_ee={w_ee}")
            print(f"  mean |e_q|_2={torch.norm(e_q, dim=1).mean().item():.4f}, "
                  f"mean |e_qd|_2={torch.norm(e_qd, dim=1).mean().item():.4f}")
            print(f"  terms(env0): pose={r_pose[0].item():.6f}, vel={r_vel[0].item():.6f}, "
                  f"ee={(r_ee[0].item() if isinstance(r_ee, torch.Tensor) else 0.0):.6f}, "
                  f"u={r_u[0].item():.6f}, du={r_du[0].item():.6f}, "
                  f"boundary={(boundary_penalty[0].item() if use_boundary else 0.0):.6f}")
            print(f"  TOTAL(env0)={total[0].item():.6f}")
            # 타깃/현재 출력
            print(f"  Target joints[0]:  {q_star[0].detach().cpu().numpy()}")
            print(f"  Current joints[0]: {q[0].detach().cpu().numpy()}")
            print(f"  Next target[0]:    {q_star_n[0].detach().cpu().numpy()}")

    # ---------------------------------------------------------
    # (7) 히스토리 (env 0만 기록 → 시각화 재사용)
    # ---------------------------------------------------------
    if "_joint_tracking_history" in globals():
        globals()["_joint_tracking_history"].append(
            (step, q_star[0].detach().cpu().numpy(), q[0].detach().cpu().numpy())
        )

    # ---------------------------------------------------------
    # (8) 에피소드 끝에서 시각화 호출 (기존 함수 사용)
    # ---------------------------------------------------------
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        if step > 0 and step % int(env.max_episode_length) == 0:
            if "_joint_tracking_history" in globals() and globals()["_joint_tracking_history"]:
                # 사용자가 이미 갖고 있는 save_joint_tracking_plot(env)를 그대로 호출
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
