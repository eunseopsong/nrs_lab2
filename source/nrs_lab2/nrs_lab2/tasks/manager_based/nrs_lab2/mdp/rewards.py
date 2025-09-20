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


# -------------------
# Reward functions
# -------------------

def joint_target_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """MSE 기반: target joints 와 현재 joints 차이"""
    q = env.scene["robot"].data.joint_pos
    T = _hdf5_trajectory.shape[0]
    idx = min(env.common_step_counter, T - 1)
    target = get_hdf5_target(env.common_step_counter).unsqueeze(0).repeat(env.num_envs, 1)
    error = torch.mean((q - target) ** 2, dim=-1)

    # ✅ 디버그 출력 (env 0만)
    if env.common_step_counter % 100 == 0:  # 매 100 step마다 출력
        current_time = env.common_step_counter * env.step_dt
        print(f"[Step {env.common_step_counter} | Time {current_time:.2f}s | HDF5 idx {idx}] "
              f"Target[0]: {target[0].cpu().numpy()} "
              f"Current[0]: {q[0].cpu().numpy()} "
              f"Error[0]: {error[0].item():.6f}")

    return error




def joint_target_tanh(env: ManagerBasedRLEnv) -> torch.Tensor:
    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env.common_step_counter).unsqueeze(0).repeat(env.num_envs, 1)
    mse = torch.mean((q - target) ** 2, dim=-1)
    return 1.0 - torch.tanh(mse)


def joint_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """속도가 너무 빠르면 패널티"""
    qd = env.scene["robot"].data.joint_vel
    return -torch.mean(qd ** 2, dim=-1)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    직전 액션과 현재 액션 차이를 줄이도록 유도
    - joint_pos_target 사용 (policy가 낸 action이 여기에 들어감)
    """
    if not hasattr(env, "_last_action"):
        env._last_action = torch.zeros_like(env.scene["robot"].data.joint_pos_target)

    current_action = env.scene["robot"].data.joint_pos_target.clone()
    diff = current_action - env._last_action
    env._last_action = current_action.detach()

    return -torch.mean(diff ** 2, dim=-1)


def q1_stability_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """q1이 0에 가까울수록 보상"""
    q = env.scene["robot"].data.joint_pos
    q1 = q[:, 0]
    return -q1**2


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
