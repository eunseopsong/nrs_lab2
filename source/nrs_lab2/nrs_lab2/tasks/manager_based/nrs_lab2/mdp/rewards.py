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
# Joint tracking reward (현재 + 미래 5-step 감쇠 포함)
# -------------------
def joint_tracking_reward(env: ManagerBasedRLEnv, gamma: float = 0.8, horizon: int = 5) -> torch.Tensor:
    # 현재 조인트 상태
    currents = env.scene["robot"].data.joint_pos[:, :6]   # [num_envs, D]
    velocities = env.scene["robot"].data.joint_vel[:, :6] # [num_envs, D]

    # future target joints (shape: [num_envs, horizon*D])
    targets_flat = get_hdf5_target_future(env, horizon=horizon)   # [num_envs, horizon*D]

    # reshape → [num_envs, horizon, D]
    num_envs, flat_dim = targets_flat.shape
    D = currents.shape[1]
    targets = targets_flat.view(num_envs, horizon, D)

    total_reward = torch.zeros(num_envs, device=env.device)

    for k in range(horizon):
        # horizon step 뒤 target
        target_k = targets[:, k, :]   # [num_envs, D]

        # 현재 상태와의 오차
        pos_err = currents - target_k
        vel_err = velocities  # target 속도 = 0 가정

        # 두 가지 reward 항목 (원래 있던 것들)
        err_sq = (pos_err**2).sum(dim=-1) + (vel_err**2).sum(dim=-1)
        err_tanh = torch.tanh(pos_err.abs().sum(dim=-1) + vel_err.abs().sum(dim=-1))

        # 감쇠율 적용
        weight = gamma**k
        total_reward += weight * ( -err_sq - err_tanh )

    return total_reward








# -------------------
# Visualization
# -------------------
_episode_counter = 0   # ✅ 전역 카운터

def save_joint_tracking_plot(env: ManagerBasedRLEnv):
    """joint target vs current trajectory 시각화 (q1~q6, 같은 번호는 같은 색)"""
    global _joint_tracking_history, _episode_counter

    if not _joint_tracking_history:
        return

    import numpy as np
    import matplotlib.pyplot as plt

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
