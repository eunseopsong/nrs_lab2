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


def joint_tracking_reward(env: ManagerBasedRLEnv, weight_pos: float = 1.0, weight_vel: float = 0.1) -> torch.Tensor:
    """Reward: penalize position + velocity error (uses future target)"""
    # robot state
    q = env.scene["robot"].data.joint_pos        # [num_envs, D]
    qd = env.scene["robot"].data.joint_vel       # [num_envs, D]

    # target state (현재 + 미래 horizon 1 step)
    targets = get_hdf5_target_future(env, horizon=2)   # [num_envs, 2*D]
    D = q.shape[1]
    q_target = targets[:, :D]        # 현재 target
    q_target_next = targets[:, D:2*D]  # +1 step target
    qd_target = (q_target_next - q_target) / env.step_dt

    # errors
    pos_error = torch.norm(q - q_target, dim=-1)
    vel_error = torch.norm(qd - qd_target, dim=-1)

    reward = -(weight_pos * pos_error + weight_vel * vel_error)

    # Debug print (optional)
    if env.common_step_counter % 10 == 0:
        print(f"[Step {env.common_step_counter}] pos_err[0]={pos_error[0].item():.4f}, vel_err[0]={vel_error[0].item():.4f}")

    return reward



# -------------------
# Visualization
# -------------------
_episode_counter = 0   # ✅ 전역 카운터

def save_joint_tracking_plot(env: ManagerBasedRLEnv):
    """joint target vs current trajectory 시각화"""
    global _joint_tracking_history, _episode_counter

    if not _joint_tracking_history:
        return

    steps, targets, currents = zip(*_joint_tracking_history)
    targets = torch.from_numpy(np.array(targets))   # ✅ 속도 개선
    currents = torch.from_numpy(np.array(currents))

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
