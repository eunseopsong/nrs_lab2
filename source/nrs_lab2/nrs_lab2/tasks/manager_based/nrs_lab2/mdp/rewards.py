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
# Joint tracking reward (현재 + 미래 5-step 감쇠 포함, 시각화 연동)
# -------------------

import matplotlib
matplotlib.use("Agg")   # ✅ headless 환경에서도 저장되게 강제

def joint_tracking_reward(env: ManagerBasedRLEnv, gamma: float = 0.9, horizon: int = 10):
    global _joint_tracking_history

    q = env.scene["robot"].data.joint_pos[:, :6]
    future_targets = get_hdf5_target_future(env, horizon=horizon)

    num_envs, D = q.shape
    horizon = min(horizon, future_targets.shape[1] // D)
    total_reward = torch.zeros(num_envs, device=env.device)

    for k in range(horizon):
        target_k = future_targets[:, k*D:(k+1)*D]
        diff = q - target_k
        rew_pos = -torch.norm(diff, dim=1)
        rew_tanh = torch.tanh(-torch.norm(diff, dim=1))
        total_reward += (gamma ** k) * (rew_pos + rew_tanh)

    # ✅ 모든 step 기록하도록 수정
    step = int(env.common_step_counter)
    target_now = future_targets[:, :D][0].detach().cpu().numpy()
    current_now = q[0].detach().cpu().numpy()
    _joint_tracking_history.append((step, target_now, current_now))

    # ✅ 1000 step마다 강제로 저장
    if step > 0 and step % 1000 == 0:
        save_joint_tracking_plot(env)

    return total_reward

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



# -----------------------------------------------------------------------------
# Behavior Cloning Model as Target Trajectory (Reset + Step 연동 버전)
# -----------------------------------------------------------------------------
import importlib
import torch
import numpy as np
import h5py

# ✅ LSTMPolicy 동적 import
try:
    lstm_module = importlib.import_module(
        "nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.lstm"
    )
    LSTMPolicy = getattr(lstm_module, "LSTMPolicy")
    print("[INFO] ✅ Successfully imported LSTMPolicy from mdp.lstm")
except Exception as e:
    print(f"[WARNING] ❌ Failed to import LSTMPolicy: {e}")
    LSTMPolicy = None


# ===============================================================
# ① Episode 시작 시 모델 로드 및 trajectory 생성
# ===============================================================
def load_bc_trajectory(env, env_ids, seq_len: int = 10):
    """
    Called at each environment reset (episode start).
    Loads model_bc.pt and generates full joint trajectory to follow.
    Compatible with Isaac Lab's EventManager.
    """
    print("\n[BC Loader] 🔄 Reloading model_bc.pt for new episode...")

    model_path = "/home/eunseop/nrs_lab2/datasets/model_bc.pt"
    data_path = "/home/eunseop/nrs_lab2/datasets/joint_recording.h5"

    if LSTMPolicy is None:
        print("[ERROR] LSTMPolicy import failed.")
        return

    # (1) Load BC model
    env.bc_model = LSTMPolicy(input_dim=6, hidden_dim=128, output_dim=6)
    env.bc_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    env.bc_model.eval()
    print(f"[BC Loader] ✅ Loaded BC model from {model_path}")

    # (2) Load dataset
    import h5py
    with h5py.File(data_path, "r") as f:
        joint_data = np.array(f["joint_positions"])
    print(f"[BC Loader] ✅ Loaded dataset: {joint_data.shape}")

    # (3) Generate predicted trajectory
    seq, preds = [], []
    for t in range(len(joint_data)):
        seq.append(joint_data[t])
        if len(seq) > seq_len:
            seq.pop(0)
        if len(seq) == seq_len:
            x = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                y = env.bc_model(x)
            preds.append(y.squeeze(0).numpy())
        else:
            preds.append(np.zeros(6))

    env._bc_full_target = np.array(preds)
    env._bc_step_counter = 0
    print(f"[BC Loader] ✅ Generated BC target trajectory: {env._bc_full_target.shape}")


# -----------------------------------------------------------------------------
# Behavior Cloning trajectory tracking reward (v13.1: interpolation + visualization)
# -----------------------------------------------------------------------------
import torch
import numpy as np
import h5py
import os
from matplotlib import pyplot as plt


def update_bc_target(env, env_ids=None):
    global _joint_tracking_history, _episode_counter

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # ---------------------------------------------------------
    # (1) Simulation parameters
    # ---------------------------------------------------------
    dt = getattr(env.sim, "dt", 1.0 / 60.0)
    decimation = getattr(env, "decimation", 2)
    episode_length_s = 60.0
    env.cfg.episode_length_s = episode_length_s
    episode_len_steps = int(episode_length_s / (dt * decimation))

    # ---------------------------------------------------------
    # (2) Trajectory loading
    # ---------------------------------------------------------
    data_path = "/home/eunseop/nrs_lab2/datasets/joint_recording.h5"
    if not hasattr(env, "_bc_full_target"):
        with h5py.File(data_path, "r") as f:
            joint_data = np.array(f["joint_positions"])
        env._bc_full_target = torch.tensor(joint_data, dtype=torch.float32, device=env.device)
        env._bc_step_counter = 0
        print(f"[INFO] Loaded HDF5 trajectory of shape {env._bc_full_target.shape}")

    if not isinstance(env._bc_full_target, torch.Tensor):
        env._bc_full_target = torch.tensor(env._bc_full_target, dtype=torch.float32, device=env.device)

    total_traj_len = env._bc_full_target.shape[0]

    # ---------------------------------------------------------
    # (3) Step indexing with linear interpolation
    # ---------------------------------------------------------
    HORIZON = 40
    GAMMA = 0.9
    SIGMA = 0.3

    current_step = env._bc_step_counter
    q_current = env.scene["robot"].data.joint_pos[env_ids, :6]
    total_reward = torch.zeros(q_current.shape[0], device=env.device)

    # 실수 인덱스 계산
    scaled_idx_f = (current_step / episode_len_steps) * total_traj_len
    idx0 = int(torch.floor(torch.tensor(scaled_idx_f)))
    idx0 = min(idx0, total_traj_len - 2)  # ✅ 안전 클램프
    idx1 = idx0 + 1
    alpha = scaled_idx_f - idx0  # 보간 비율 (0~1)

    # ✅ 디버깅 출력 복원
    if current_step % 100 == 0 or current_step < 10:
        print(f"[DEBUG] Step {current_step:04d}/{episode_len_steps} | scaled_idx_f={scaled_idx_f:.2f}, idx0={idx0}, idx1={idx1}, α={alpha:.3f}")

    # 선형 보간된 현재 target joint
    q_target_interp = (1 - alpha) * env._bc_full_target[idx0] + alpha * env._bc_full_target[idx1]


    # ---------------------------------------------------------
    # (4) Future target sampling (interpolated sequence)
    # ---------------------------------------------------------
    D = q_current.shape[1]
    q_target_future = []

    for k in range(HORIZON):
        scaled_f_k = scaled_idx_f + k
        idx0_k = int(torch.floor(torch.tensor(scaled_f_k)))
        idx0_k = min(idx0_k, total_traj_len - 2)  # ✅ out-of-bound 방지
        idx1_k = idx0_k + 1
        alpha_k = scaled_f_k - idx0_k
        q_interp_k = (1 - alpha_k) * env._bc_full_target[idx0_k] + alpha_k * env._bc_full_target[idx1_k]
        q_target_future.append(q_interp_k.unsqueeze(0))

    q_target_future = torch.cat(q_target_future, dim=0)


    # ---------------------------------------------------------
    # (5) Reward accumulation
    # ---------------------------------------------------------
    for k in range(HORIZON):
        q_target_h = q_target_future[k]
        diff = q_current - q_target_h
        error = torch.norm(diff, dim=1)

        rew_linear = -error
        rew_tanh = 1 - torch.tanh(error / SIGMA)

        total_reward += (GAMMA ** k) * (rew_linear + rew_tanh)

    total_reward /= HORIZON

    # ---------------------------------------------------------
    # (6) Temporal smoothing
    # ---------------------------------------------------------
    alpha_smooth = 0.5
    if hasattr(env, "_prev_reward"):
        total_reward = alpha_smooth * env._prev_reward + (1 - alpha_smooth) * total_reward
    env._prev_reward = total_reward.clone()

    # ---------------------------------------------------------
    # (7) Logging
    # ---------------------------------------------------------
    if current_step % 100 == 0:
        mean_r = total_reward.mean().item()
        print(f"[BC Tracking v13.1] Step {current_step:05d} | H={HORIZON}, γ={GAMMA}, mean_r={mean_r:.4f}")

        _joint_tracking_history.append((
            current_step,
            q_target_interp.detach().cpu().numpy(),
            q_current[0].detach().cpu().numpy()
        ))

    # ---------------------------------------------------------
    # (8) Episode end & visualization
    # ---------------------------------------------------------
    env._bc_step_counter += 1
    if env._bc_step_counter >= episode_len_steps:
        print("[BC Loader] 🔁 Reloading BC trajectory for new episode...")

        # ✅ 시각화 (Target vs Current)
        if _joint_tracking_history:
            steps, targets, currents = zip(*_joint_tracking_history)
            targets = np.array(targets)
            currents = np.array(currents)

            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab10.colors
            for j in range(targets.shape[1]):
                c = colors[j % len(colors)]
                plt.plot(steps, targets[:, j], "--", label=f"Target q{j+1}", color=c, linewidth=1.2)
                plt.plot(steps, currents[:, j], "-", label=f"Current q{j+1}", color=c, linewidth=2.0)
            plt.xlabel("Step")
            plt.ylabel("Joint (rad)")
            plt.title("Joint Tracking (Target vs Current)")
            plt.legend(ncol=2, fontsize=8)
            plt.grid(True)

            save_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"bc_tracking_episode{_episode_counter}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[INFO] Saved BC tracking plot to {save_path}")

            _episode_counter += 1
            _joint_tracking_history.clear()

        env._bc_step_counter = 0

    return total_reward
