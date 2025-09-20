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


# ---------------------------
# Termination: reached_end
# ---------------------------
def reached_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    종료 조건: HDF5 trajectory 마지막 값에 도달하면 True
    """
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")
    T = _hdf5_trajectory.shape[0]
    done = env.common_step_counter >= (T - 1)
    return torch.tensor([done] * env.num_envs, dtype=torch.bool, device=env.device)
