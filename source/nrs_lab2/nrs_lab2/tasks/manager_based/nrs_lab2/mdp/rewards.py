# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
import h5py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_hdf5_trajectory = None
_prev_error = None   # convergence 보상 계산용


def load_hdf5_trajectory(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "joint_positions"):
    """HDF5 trajectory 데이터를 로드 (reset 시 1회 호출)"""
    global _hdf5_trajectory, _prev_error
    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5: {dataset_key} not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # [T, D]
    _hdf5_trajectory = torch.tensor(data, dtype=torch.float32, device=env.device)
    _prev_error = None   # reset 시 초기화


def get_hdf5_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")

    T = _hdf5_trajectory.shape[0]
    E = env.max_episode_length
    step = env.episode_length_buf[0].item()
    idx = min(int(step / E * T), T - 1)

    return _hdf5_trajectory[idx]


# -------------------
# Reward functions
# -------------------

def joint_target_error_weighted(env: ManagerBasedRLEnv, weights=None) -> torch.Tensor:
    """Joint-wise weighted tracking error"""
    if weights is None:
        weights = [2.0, 2.0, 2.0, 5.0, 1.0, 1.0]  # q4 강조

    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env).unsqueeze(0).repeat(env.num_envs, 1)

    diff = q - target
    weighted_error = torch.sum(torch.tensor(weights, device=env.device) * (diff ** 2), dim=-1)

    # 디버깅 출력
    if env.common_step_counter % 10 == 0:
        current_time = env.common_step_counter * env.step_dt
        print(f"[Step {env.common_step_counter} | Time {current_time:.2f}s] "
              f"Target[0]: {target[0].cpu().numpy()} "
              f"Current[0]: {q[0].cpu().numpy()} "
              f"WeightedError[0]: {weighted_error[0].item():.6f}")

    return -weighted_error


# def q4_active_tracking(env: ManagerBasedRLEnv, threshold: float = 1e-3) -> torch.Tensor:
#     """q4 joint가 멈춰있지 않고 target을 따라 움직이도록 패널티 적용"""
#     qd = env.scene["robot"].data.joint_vel
#     q4_vel = torch.abs(qd[:, 3])

#     penalty = torch.where(q4_vel < threshold, -2.0 * torch.ones_like(q4_vel), torch.zeros_like(q4_vel))

#     if env.common_step_counter % 100 == 0:
#         print(f"[Step {env.common_step_counter}] q4_vel[0]: {q4_vel[0].item():.6f}, Penalty[0]: {penalty[0].item():.6f}")

#     return penalty


def fast_convergence_reward(env: ManagerBasedRLEnv, weights=None) -> torch.Tensor:
    """
    에러 감소 속도 기반 보상
    - 이전 error 대비 감소하면 보상 부여
    """
    global _prev_error
    if weights is None:
        weights = [2.0, 2.0, 2.0, 5.0, 1.0, 1.0]

    q = env.scene["robot"].data.joint_pos
    target = get_hdf5_target(env).unsqueeze(0).repeat(env.num_envs, 1)
    diff = q - target
    weighted_error = torch.sum(torch.tensor(weights, device=env.device) * (diff ** 2), dim=-1)

    if _prev_error is None:
        _prev_error = weighted_error.clone()
        return torch.zeros_like(weighted_error)

    delta = _prev_error - weighted_error
    _prev_error = weighted_error.clone()

    # error 줄어든 경우만 보상 (delta > 0)
    reward = torch.clamp(delta, min=0.0)

    if env.common_step_counter % 100 == 0:
        print(f"[Step {env.common_step_counter}] FastConvergenceReward[0]: {reward[0].item():.6f}")

    return reward


# -------------------
# Termination function
# -------------------

def reached_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_trajectory.shape[0]
    return torch.tensor(env.common_step_counter >= (T - 1),
                        dtype=torch.bool, device=env.device).repeat(env.num_envs)
