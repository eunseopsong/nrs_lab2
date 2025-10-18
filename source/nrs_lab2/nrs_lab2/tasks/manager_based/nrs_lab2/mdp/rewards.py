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


# ---------------------------------------------
# Joint tracking reward (민감도 향상판)
# - 범위 정규화 + Huber/가우시안 커널(선택/혼합)
# - 속도 추종, 액션/Δ액션 패널티(per-dim mean), 프리뷰 보조항
# - 10 step마다 디버그: current/target/error 및 각 보상 항목
# - 에피소드 경계에서 save_joint_tracking_plot(env) 호출(외부 정의)
# ---------------------------------------------

import torch

# 전역 캐시(이미 있으면 재사용)
try:
    _joint_bounds
except NameError:
    _joint_bounds = None

try:
    _joint_tracking_history
except NameError:
    _joint_tracking_history = []

def _huber(x: torch.Tensor, delta: float):
    ax = torch.abs(x)
    quad = 0.5 * (x ** 2)
    lin  = delta * (ax - 0.5 * delta)
    return torch.where(ax <= delta, quad, lin)

def joint_tracking_reward(
    env: "ManagerBasedRLEnv",
    # ----- 기본 튜닝값 -----
    delta: float = 0.06,          # 위치 Huber δ  (작을수록 민감)
    delta_v: float = 0.15,        # 속도 Huber δ
    w_pos: float = 0.6,           # 위치/속도 가중(합=1 권장)
    w_vel: float = 0.4,
    lam_u: float = 3e-3,          # 액션 L2 패널티(평균)
    lam_du: float = 7e-3,         # Δ액션 L2 패널티(평균)
    beta_prog: float = 0.03,      # 프리뷰(코사인) 보조항
    # ----- 민감도/스케일 옵션 -----
    pos_kernel: str = "mix",      # "huber" | "gauss" | "mix"
    pos_gain: float = 3.0,        # Huber penalty gain (1 - gain*pen)
    sigma_pos: float = 0.15,      # 가우시안 σ (정규화 e 기준)
    bounds_mode: str = "percentile"  # "minmax" | "percentile"
):
    """
    Returns:
        r_total: [num_envs] tensor
    Notes:
        - _joint_tracking_history 에 (step, target[6], current[6]) 저장
        - 에피소드 길이마다 save_joint_tracking_plot(env) 호출 (외부 함수)
    """
    global _joint_bounds, _joint_tracking_history

    robot = env.scene["robot"]
    q  = robot.data.joint_pos[:, :6]   # [E,6]
    qd = robot.data.joint_vel[:, :6]
    E, D = q.shape

    # ----- 타겟 (t, t+1 프리뷰) -----
    fut = get_hdf5_target_future(env, horizon=2)   # [E, 12]
    q_star     = fut[:, :D]
    q_star_nxt = fut[:, D:2*D] if fut.shape[1] >= 2*D else q_star

    # ----- 관절 가중치 -----
    wj = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 0.5], device=env.device).reshape(1, D)

    # ----- joint bounds (퍼센타일 권장) -----
    rng_eps = 1e-3
    if _joint_bounds is None:
        try:
            from builtins import _hdf5_trajectory
            import numpy as np
            if bounds_mode == "percentile":
                p2  = np.percentile(_hdf5_trajectory,  2, axis=0)
                p98 = np.percentile(_hdf5_trajectory, 98, axis=0)
                joint_min = torch.tensor(p2[:D],  device=env.device, dtype=torch.float32)
                joint_max = torch.tensor(p98[:D], device=env.device, dtype=torch.float32)
            else:  # "minmax"
                joint_min = torch.tensor(np.min(_hdf5_trajectory, axis=0)[:D],  device=env.device, dtype=torch.float32)
                joint_max = torch.tensor(np.max(_hdf5_trajectory, axis=0)[:D],  device=env.device, dtype=torch.float32)
        except Exception:
            q0 = q.detach()
            joint_min = (q0.min(dim=0).values - 1.0).to(env.device)
            joint_max = (q0.max(dim=0).values + 1.0).to(env.device)
        _joint_bounds = (joint_min, joint_max)

    joint_min, joint_max = _joint_bounds
    rng = (joint_max - joint_min).clamp_min(rng_eps)  # [6]

    # ----- 정규화 오차 -----
    e  = (q - q_star) / rng   # 위치 오차(정규화)
    dt  = getattr(env.sim, "dt", 1.0 / 60.0)
    dec = getattr(env, "decimation", 2)
    dt_eff = dt * dec
    qd_star = (q_star_nxt - q_star) / max(dt_eff, 1e-6)
    ev = (qd - qd_star) / 2.0  # 속도 스케일 완화

    # ----- 위치 보상: Huber / Gaussian / Mix -----
    pen_huber = (_huber(e, delta) * wj).mean(dim=1)          # [E]
    e_norm2   = ((e * wj).pow(2).sum(dim=1)).clamp_min(1e-12)
    rew_gauss = torch.exp(- e_norm2 / (sigma_pos ** 2))      # [E]

    if pos_kernel == "huber":
        r_pos = 1.0 - pos_gain * pen_huber
    elif pos_kernel == "gauss":
        r_pos = rew_gauss
    else:  # "mix"
        r_pos = 0.5 * (1.0 - pos_gain * pen_huber) + 0.5 * rew_gauss

    r_pos = torch.clamp(r_pos, min=-1.0, max=1.2)

    # ----- 속도 보상: Huber -----
    r_vel = 1.0 - _huber(ev, delta_v).mean(dim=1)  # [E]

    # ----- 액션/Δ액션 패널티 (per-dim mean) -----
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        a = env.action_manager.action
    elif hasattr(env, "actions"):
        a = env.actions
    else:
        a = torch.zeros((E, D), device=env.device)

    if not hasattr(env, "_prev_action"):
        env._prev_action = torch.zeros_like(a)

    r_u  = -(a.square().mean(dim=1)) * lam_u
    r_du = -((a - env._prev_action).square().mean(dim=1)) * lam_du
    env._prev_action = a.clone()

    # ----- 프리뷰 보조항: 방향 코사인 -----
    prog = torch.zeros(E, device=env.device)
    if fut.shape[1] >= 2 * D:
        dir_tar = (q_star_nxt - q).detach()
        if not hasattr(env, "_q_prev"):
            env._q_prev = q.clone()
        dir_act = (q - env._q_prev).detach()
        env._q_prev = q.clone()

        num = torch.sum(dir_tar * dir_act, dim=1)
        den = dir_tar.norm(dim=1) * dir_act.norm(dim=1) + 1e-6
        cos = num / den
        prog = beta_prog * (cos + 1.0) * 0.5  # [0, beta_prog]

    # ----- 총합 보상 -----
    r_total = (w_pos * r_pos + w_vel * r_vel) + r_u + r_du + prog
    r_total = torch.clamp(r_total, min=-5.0, max=+5.0)

    # ----- 히스토리/플롯 -----
    step = int(getattr(env, "common_step_counter", 0))
    _joint_tracking_history.append(
        (step, q_star[0].detach().cpu().numpy(), q[0].detach().cpu().numpy())
    )
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        if step > 0 and step % int(env.max_episode_length) == 0:
            if _joint_tracking_history:
                # 외부에 동일 이름 함수가 정의되어 있어야 함
                save_joint_tracking_plot(env)

    # ----- 디버그(10 step마다, env 0) -----
    if step % 10 == 0:
        e_vec = (q[0] - q_star[0])
        print(f"\n[Step {step}]")
        print(f"  Target joints[0]: {q_star[0].detach().cpu().numpy()}")
        print(f"  Current joints[0]: {q[0].detach().cpu().numpy()}")
        print(f"  Error (q - q* )[0]: {e_vec.detach().cpu().numpy()}")
        print(f"  ‖Error‖_2[0]: {torch.norm(e_vec).item():.6f}")
        print(f"  pos_kernel={pos_kernel}, bounds={bounds_mode}, pos_gain={pos_gain}, sigma_pos={sigma_pos}")
        print(f"  mean|e|_norm={e[0].abs().mean().item():.4f}, L2^2(e_norm)={e_norm2[0].item():.4f}")
        print("  Reward terms (env 0):")
        print(f"    r_pos={r_pos[0].item():.6f}, r_vel={r_vel[0].item():.6f}, "
              f"r_u={r_u[0].item():.6f}, r_du={r_du[0].item():.6f}, prog={prog[0].item():.6f}")
        print(f"    TOTAL={r_total[0].item():.6f}")

    return r_total





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
