# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
import h5py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_hdf5_trajectory = None


def load_hdf5_trajectory(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "joint_positions"):
    """HDF5 trajectory 데이터를 로드 (reset 시 1회 호출)"""
    global _hdf5_trajectory
    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5: {dataset_key} not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # [T, D]
    _hdf5_trajectory = torch.tensor(data, dtype=torch.float32, device=env.device)


def get_hdf5_target(t: int) -> torch.Tensor:
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")
    T = _hdf5_trajectory.shape[0]
    idx = min(t, T - 1)
    return _hdf5_trajectory[idx]


def joint_target_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env.common_step_counter).unsqueeze(0).repeat(env.num_envs, 1)
    return torch.mean((q - target) ** 2, dim=-1)


def joint_target_tanh(env: ManagerBasedRLEnv) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env.common_step_counter).unsqueeze(0).repeat(env.num_envs, 1)
    mse = torch.mean((q - target) ** 2, dim=-1)
    return 1.0 - torch.tanh(mse)


def joint_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """속도 제곱 페널티: 로봇이 과도하게 빠르게 움직이지 않도록 억제"""
    qd = env.scene["robot"].data.joint_vel
    return -torch.mean(qd ** 2, dim=-1)


def reached_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    """HDF5 trajectory 끝에 도달하면 종료"""
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        # 모든 env에 대해 False 반환
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_trajectory.shape[0]
    # step counter가 trajectory 끝을 넘으면 True
    done = env.common_step_counter >= (T - 1)
    return torch.full((env.num_envs,), done, dtype=torch.bool, device=env.device)

