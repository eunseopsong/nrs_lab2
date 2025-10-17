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
# Joint tracking reward (Gaussian kernel flattening)
# + Convergence boost term (meta reward)
# + Smoothed boundary penalty
# -------------------

import matplotlib
matplotlib.use("Agg")

_prev_total_reward = None  # 전역 변수로 이전 step reward 저장
_joint_bounds = None       # 각 joint의 [min, max] 저장용


def joint_tracking_reward(env: ManagerBasedRLEnv, sigma: float = 2.0, alpha: float = 3.0):
    global _joint_tracking_history, _episode_counter, _prev_total_reward, _joint_bounds

    # ------------------------------
    # (0) 현재 상태
    # ------------------------------
    q = env.scene["robot"].data.joint_pos[:, :6]
    qd = env.scene["robot"].data.joint_vel[:, :6]
    future_targets = get_hdf5_target_future(env, horizon=2)  # t, t+1만 필요

    num_envs, D = q.shape
    total_reward = torch.zeros(num_envs, device=env.device)

    # ------------------------------
    # (1) Target 전체 범위로부터 boundary 설정 (episode 초기화 시 1회)
    # ------------------------------
    if _joint_bounds is None:
        path = "/home/eunseop/nrs_lab2/datasets/joint_recording.h5"
        import h5py
        with h5py.File(path, "r") as f:
            joint_data = f["joint_positions"][:]  # ✅ key 이름 수정
        joint_min = torch.tensor(joint_data.min(axis=0), device=env.device, dtype=torch.float32)
        joint_max = torch.tensor(joint_data.max(axis=0), device=env.device, dtype=torch.float32)
        _joint_bounds = (joint_min, joint_max)
        print(f"[INFO] Joint boundaries set: min={joint_min.cpu().numpy()}, max={joint_max.cpu().numpy()}")

    joint_min, joint_max = _joint_bounds

    # ------------------------------
    # (2) Base Reward 계산 (exp 커널 평탄화 + Integral Error Term)
    # ------------------------------
    joint_weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 0.5], device=env.device)
    vel_weight = 0.3
    pos_weight = 0.7

    next_target = future_targets[:, D:2*D] if future_targets.shape[1] >= 2*D else future_targets[:, :D]
    diff = q - next_target

    # Position reward
    weighted_sq = joint_weights * (diff ** 2)
    sq_norm = torch.sum(weighted_sq, dim=1)
    rew_pos = torch.exp(-sq_norm / (sigma ** 2))

    # Velocity reward
    # diff_dot = qd
    # vel_norm = torch.norm(diff_dot, dim=1)
    # rew_vel = torch.exp(-vel_norm / (sigma ** 2))

    # Base reward (7:3 비율)
    # base_reward = pos_weight * rew_pos + vel_weight * rew_vel
    base_reward = rew_pos  # velocity 항 제거

    # ------------------------------
    # (2-1) Integral Error Term (steady-state error 제거)
    # ------------------------------
    # if not hasattr(env, "_integral_error"):
    #     env._integral_error = torch.zeros_like(base_reward)

    # 누적 오차 업데이트 (지수 감쇠)
    # beta = 0.98  # 0.95~0.99 권장 (감쇠율)
    # env._integral_error = beta * env._integral_error + torch.norm(diff, dim=1)

    # integral penalty (steady-state error 방지)
    # k_i = 0.2  # integral 강도 (0.1~0.3 권장)
    # integral_penalty = k_i * env._integral_error

    # 보상에서 감산
    # base_reward = base_reward - integral_penalty


    # ------------------------------
    # (3) Convergence Boost Reward 추가
    # ------------------------------
    boost_reward = reward_convergence_boost(env, base_reward, alpha)

    # ------------------------------
    # (4) 완화된 Boundary Penalty
    # ------------------------------
    # margin을 두어 약간의 초과는 허용
    margin = 0.2
    overflow_low = torch.clamp(joint_min - q, min=0.0)
    overflow_high = torch.clamp(q - joint_max, min=0.0)
    overflow = overflow_low + overflow_high

    # overflow 합산 (joint별로 weighted 평균)
    overflow_norm = torch.sum(overflow * joint_weights, dim=1)

    # ✅ 완화된 penalty (k값 ↓)
    k = 6.0
    # ✅ boundary_penalty: 정상(0), 벗어날수록 1에 가까움
    boundary_penalty = 1.0 - torch.exp(-k * overflow_norm)

    # 감산형 penalty 적용 (reward 감소)
    total_reward = (base_reward + boost_reward) * (1.0 - boundary_penalty)
    total_reward = torch.clamp(total_reward, min=0.0)


    # ------------------------------
    # (5) 디버깅 출력 (매 100 step)
    # ------------------------------
    step = int(env.common_step_counter)
    if step % 100 == 0:
        err_norm = torch.norm(diff[0]).item()
        # int_err = env._integral_error[0].item() if hasattr(env, "_integral_error") else 0.0

        print(f"\n[Step {step}]")
        print(f"  Target joints : {next_target[0].detach().cpu().numpy()}")
        print(f"  Current joints: {q[0].detach().cpu().numpy()}")
        print(f"  Error (‖q - q*‖): {err_norm:.6f}")
        # print(f"  Integral error (Σ‖e‖): {int_err:.6f}")
        # print(f"  Reward_pos: {rew_pos[0].item():.6f}, Reward_vel: {rew_vel[0].item():.6f}")
        print(f"  Reward_pos: {rew_pos[0].item():.6f}")
        print(f"  Base_total: {base_reward[0].item():.6f}, Boost: {boost_reward[0].item():.6f}")
        print(f"  Penalty: {1.0 - boundary_penalty[0].item():.6f}, Final Reward: {total_reward[0].item():.6f}")

    # ------------------------------
    # (6) History 저장
    # ------------------------------
    target_now = next_target[0].detach().cpu().numpy()
    current_now = q[0].detach().cpu().numpy()
    _joint_tracking_history.append((step, target_now, current_now))

    # ------------------------------
    # (7) Episode 종료 시 시각화
    # ------------------------------
    if hasattr(env, "max_episode_length") and env.max_episode_length > 0:
        if step > 0 and step % int(env.max_episode_length) == 0:
            if _joint_tracking_history:
                save_joint_tracking_plot(env)

    return total_reward


# --------------------------------
# Reward improvement (meta reward)
# --------------------------------
def reward_convergence_boost(env, current_reward: torch.Tensor, alpha: float = 3.0):
    """
    강화된 수렴 보상 (Reward Improvement Term)
    - 이전 step보다 reward_pos가 커지면 positive boost
    - 감소하면 penalty 적용
    - alpha: 보상 강도 (0.5~3.0 권장)
    - ✅ episode 초기화 시에도 _prev_total_reward 유지
    """
    global _prev_total_reward

    # IsaacLab의 env.reset() 시점에서도 유지되도록 None 체크 완화
    if _prev_total_reward is None or torch.isnan(current_reward).any():
        _prev_total_reward = current_reward.clone()
        return torch.zeros_like(current_reward)

    # reward 변화량 계산
    reward_delta = current_reward - _prev_total_reward
    reward_boost = alpha * torch.clamp(torch.tanh(reward_delta), min=0.0)  # ✅ positive delta만 boost

    # ✅ episode reset 시에도 유지되도록 단순 clone이 아닌 EMA 형태로 누적
    beta = 0.98  # 0.95~0.99 권장
    _prev_total_reward = beta * _prev_total_reward + (1 - beta) * current_reward

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
# Visualization
# -------------------
_episode_counter = 0   # ✅ 전역 카운터

def save_joint_tracking_plot(env: ManagerBasedRLEnv):
    """joint target vs current trajectory 시각화 (q1~q6, 같은 번호는 같은 색)"""
    global _joint_tracking_history, _episode_counter

    if not _joint_tracking_history:
        return

    steps, targets, currents = zip(*_joint_tracking_history)
    targets = torch.from_numpy(np.array(targets))   # ✅ 속도 개선
    currents = torch.from_numpy(np.array(currents))

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # tab10 팔레트 (10개 색상)

    for j in range(targets.shape[1]):
        color = colors[j % len(colors)]
        # q1~q6로 라벨링
        plt.plot(steps, targets[:, j], "--", label=f"Target q{j+1}", color=color, linewidth=1.2)
        plt.plot(steps, currents[:, j], "-", label=f"Current q{j+1}", color=color, linewidth=2.0)

    plt.xlabel("Step")
    plt.ylabel("Joint Value (rad)")
    plt.title("Joint Tracking (Target vs Current)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)

    # 저장 경로
    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
    os.makedirs(save_dir, exist_ok=True)

    # ✅ episode_index 대신 전역 카운터 사용
    save_path = os.path.join(save_dir, f"joint_tracking_episode{_episode_counter}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved joint tracking plot to {save_path}")

    _episode_counter += 1        # 다음 episode 번호 증가
    _joint_tracking_history = [] # 다음 episode를 위해 초기화





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
