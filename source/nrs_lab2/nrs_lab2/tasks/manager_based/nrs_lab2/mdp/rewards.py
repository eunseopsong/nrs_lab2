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
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_hdf5_trajectory = None
_joint_tracking_history = []   # (step, target, current) 기록용


# -------------------
# Load target trajectory (HDF5)
# -------------------
def load_hdf5_trajectory(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "joint_positions"):
    """HDF5 trajectory 데이터를 로드 (reset 시 1회 호출)"""
    global _hdf5_trajectory, _joint_tracking_history
    import h5py
    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5: {dataset_key} not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # [T, D]
    _hdf5_trajectory = torch.tensor(data, dtype=torch.float32, device=env.device)

    # reset history
    _joint_tracking_history = []


def get_hdf5_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 step에 해당하는 target joint 반환"""
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
# Visualization
# -------------------
def save_joint_tracking_plot(env: ManagerBasedRLEnv):
    """joint target vs current trajectory 시각화"""
    global _joint_tracking_history

    if not _joint_tracking_history:
        return

    steps, targets, currents = zip(*_joint_tracking_history)
    targets = torch.tensor(targets)
    currents = torch.tensor(currents)

    plt.figure(figsize=(10, 6))
    for j in range(targets.shape[1]):
        plt.plot(steps, targets[:, j], "--", label=f"Target q{j}")
        plt.plot(steps, currents[:, j], "-", label=f"Current q{j}")

    plt.xlabel("Step")
    plt.ylabel("Joint Value (rad)")
    plt.title("Joint Tracking (Target vs Current)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)

    # 저장 경로
    save_dir = os.path.expanduser("~/nrs_lab2/outputs/png")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"joint_tracking_step{env.common_step_counter}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved joint tracking plot to {save_path}")


# -------------------
# Termination function
# -------------------
def reached_end(env: ManagerBasedRLEnv, command_name=None, asset_cfg=None) -> torch.Tensor:
    """Trajectory 끝에 도달하면 종료"""
    global _hdf5_trajectory
    if _hdf5_trajectory is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    T = _hdf5_trajectory.shape[0]
    return torch.tensor(env.common_step_counter >= (T - 1),
                        dtype=torch.bool, device=env.device).repeat(env.num_envs)
