# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
import h5py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -------------------
# Global buffer
# -------------------
_hdf5_trajectory = None


# -------------------
# HDF5 trajectory loader
# -------------------
def load_hdf5_trajectory(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "joint_positions"):
    """HDF5 trajectory 데이터를 로드 (reset 시 1회 호출)"""
    global _hdf5_trajectory
    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5: {dataset_key} not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # [T, D]
    _hdf5_trajectory = torch.tensor(data, dtype=torch.float32, device=env.device)


def get_hdf5_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 episode step에 해당하는 HDF5 target joint 값 반환"""
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")

    T = _hdf5_trajectory.shape[0]      # HDF5 길이
    E = env.max_episode_length         # episode step 수 (예: 3600)

    # episode 내부 step counter 사용 (reset 시 0으로 돌아감)
    step = env.episode_length_buf[0].item()

    # 🔑 HDF5 인덱스를 episode 진행도에 맞춰 스케일링
    idx = min(int(step / E * T), T - 1)

    return _hdf5_trajectory[idx]


# -------------------
# Reward functions
# -------------------
def joint_target_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """목표 joint position tracking 오차 (MSE 기반)"""
    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env).unsqueeze(0).repeat(env.num_envs, 1)
    error = torch.mean((q - target) ** 2, dim=-1)

    # Debug 출력
    if env.common_step_counter % 100 == 0:
        current_time = env.common_step_counter * env.step_dt
        print(f"[Step {env.common_step_counter} | Time {current_time:.2f}s] "
              f"Target[0]: {target[0].cpu().numpy()} "
              f"Current[0]: {q[0].cpu().numpy()} "
              f"Error[0]: {error[0].item():.6f}")

    # tracking은 보상이므로 -error 반환
    return -error


def joint_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """속도가 너무 빠르면 패널티 (학습 안정화를 위한 smoothness 역할)"""
    qd = env.scene["robot"].data.joint_vel
    return -0.01 * torch.mean(qd ** 2, dim=-1)   # 🔑 scale 낮게


def q1_stability_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """q1이 0에 가까울수록 보상"""
    q = env.scene["robot"].data.joint_pos
    q1 = q[:, 0]
    return -0.1 * q1**2   # 🔑 tracking 보조 term


# -------------------
# Termination function
# -------------------
def reached_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    """HDF5 trajectory 끝에 도달하면 종료"""
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_trajectory.shape[0]
    return torch.tensor(env.common_step_counter >= (T - 1),
                        dtype=torch.bool, device=env.device).repeat(env.num_envs)
