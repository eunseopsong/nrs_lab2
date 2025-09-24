# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for UR10e spindle environment.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv

# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_hdf5_trajectory = None
_step_idx = 0


# ------------------------------------------------------
# HDF5 trajectory loader
# ------------------------------------------------------
def load_hdf5_trajectory(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "joint_positions"):
    """HDF5 trajectory 데이터를 로드 (reset 시 1회 호출)"""
    global _hdf5_trajectory, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5: {dataset_key} not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # [T, D]

    _hdf5_trajectory = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 trajectory of shape {_hdf5_trajectory.shape}")


# ------------------------------------------------------
# Observation: next target joints
# ------------------------------------------------------
def get_hdf5_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 step에 해당하는 target joint 반환"""
    global _hdf5_trajectory, _step_idx
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")

    T = _hdf5_trajectory.shape[0]
    E = env.max_episode_length
    step = env.episode_length_buf[0].item()
    idx = min(int(step / E * T), T - 1)

    return _hdf5_trajectory[idx]

# ------------------------------------------------------
# Observation: future target joints
# ------------------------------------------------------
def get_hdf5_target_future(env: ManagerBasedRLEnv, horizon: int = 5) -> torch.Tensor:
    global _hdf5_trajectory, _step_idx
    if _hdf5_trajectory is None:
        # dummy: [num_envs, horizon * D]
        D = env.scene["robot"].num_joints
        return torch.zeros((env.num_envs, horizon * D), device=env.device, dtype=torch.float32)

    T, D = _hdf5_trajectory.shape
    step = env.episode_length_buf[0].item()
    E = env.max_episode_length
    idx = min(int(step / E * T), T - 1)

    # horizon 만큼 future target 뽑기
    future_idx = torch.clamp(torch.arange(idx, idx + horizon), max=T - 1)

    # shape (horizon, D) → (1, horizon * D) → (num_envs, horizon * D)
    future_targets = _hdf5_trajectory[future_idx].reshape(1, horizon * D)
    return future_targets.repeat(env.num_envs, 1)

