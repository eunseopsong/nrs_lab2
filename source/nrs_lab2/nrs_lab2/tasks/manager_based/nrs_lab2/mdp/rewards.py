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


def get_hdf5_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")

    T = _hdf5_trajectory.shape[0]      # HDF5 길이
    E = env.max_episode_length         # episode step 수

    # episode 내부 step counter 사용 (reset 시 0으로 돌아감)
    step = env.episode_length_buf[0].item()

    # 🔑 HDF5 인덱스를 episode 진행도에 맞춰 스케일링
    idx = min(int(step / E * T), T - 1)

    return _hdf5_trajectory[idx]


# -------------------
# Reward functions
# -------------------

def joint_target_error_strict(env: ManagerBasedRLEnv, scale: float = 50.0) -> torch.Tensor:
    """
    각 joint별 target과 현재값 차이가 0에 가까울수록 큰 보상
    - 조인트별로 exp(-scale * (diff^2)) 계산 후 평균 사용
    - 디버깅 출력은 기존 형식 유지
    """
    q = env.scene["robot"].data.joint_pos  # [num_envs, num_joints]
    target = get_hdf5_target(env).unsqueeze(0).repeat(env.num_envs, 1)  # [num_envs, num_joints]

    # per-joint reward (exp shaping)
    diffs = q - target
    mse = torch.mean(diffs ** 2, dim=-1)  # 기존 MSE도 계산 (출력용)
    rewards = torch.exp(-scale * (diffs ** 2))  # [num_envs, num_joints]

    # 전체 reward = 평균
    reward = torch.mean(rewards, dim=-1)

    # ✅ 디버깅 출력 (원래 형식 유지)
    if env.common_step_counter % 100 == 0:
        current_time = env.common_step_counter * env.step_dt
        print(f"[Step {env.common_step_counter} | Time {current_time:.2f}s] "
              f"Target[0]: {target[0].cpu().numpy()} "
              f"Current[0]: {q[0].cpu().numpy()} "
              f"MSE[0]: {mse[0].item():.6f}, Reward[0]: {reward[0].item():.6f}")

    return reward


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
