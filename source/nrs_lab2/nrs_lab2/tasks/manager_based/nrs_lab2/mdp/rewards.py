# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import math
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -------------------------------
# V2: 왕복하는 joint target (start <-> goal)
# -------------------------------
START_JOINTS = torch.tensor(
    [0.0, -0.785, -0.785, -1.57, 1.57, 0.0],  # 시작 포즈
    dtype=torch.float32,
)
GOAL_JOINTS = torch.tensor(
    [0.0, -1.57, -1.57, -1.57, 1.57, 0.0],  # 끝 포즈
    dtype=torch.float32,
)

def _get_robot_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 로봇 조인트 각도 [num_envs, dof]"""
    robot = env.scene["robot"]  # Articulation
    return robot.data.joint_pos  # [N, D]

def _moving_target(env, num_joints: int):
    """주기적으로 start~goal joint를 왕복"""
    t = torch.tensor(env.common_step_counter, device=env.device, dtype=torch.float32)  # [scalar]
    episode_length = torch.tensor(env.max_episode_length, device=env.device, dtype=torch.float32)

    # start ~ goal 정의
    start = torch.tensor([0.0, 0.0, 0.0, -1.57, 1.57, 0.0], device=env.device)
    goal  = torch.tensor([0.0, -1.57, -1.57, -1.57, 1.57, 0.0], device=env.device)

    # 주기 (한 episode에서 몇 번 왕복할지)
    num_cycles = 2.0   # 2번 왕복
    omega = 2 * math.pi * num_cycles / episode_length

    # sin 기반 주기적 타겟
    alpha = 0.5 * (1.0 + torch.sin(omega * t))  # [0,1] 사이에서 왕복
    target = start + (goal - start) * alpha

    return target.unsqueeze(0).repeat(env.num_envs, 1)  # [N, D]


# ------------------------------------------------------
# Reward functions
# ------------------------------------------------------
def joint_target_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """MSE 기반 에러 (음수 가중치로 페널티로 쓰기 좋음)"""
    q = _get_robot_joint_pos(env)                  # [N, D]
    target = _moving_target(env, q.shape[1])       # [N, D]
    return torch.mean((q - target) ** 2, dim=-1)   # [N]

def joint_target_tanh(env: ManagerBasedRLEnv) -> torch.Tensor:
    """tanh 기반 보상 (값이 0~1, 가중치는 양수로 쓰는 게 일반적)"""
    q = _get_robot_joint_pos(env)                  # [N, D]
    target = _moving_target(env, q.shape[1])       # [N, D]
    mse = torch.mean((q - target) ** 2, dim=-1)    # [N]
    return 1.0 - torch.tanh(mse)                   # [N]
