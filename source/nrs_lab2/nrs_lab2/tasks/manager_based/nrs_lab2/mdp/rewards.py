# SPDX-License-Identifier: BSD-3-Clause
"""
Reward functions for UR10e joint tracking
- target_joints: 코드 내 상수로 정의
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -------------------------------
# Target joint values [rad]
# -------------------------------
TARGET_JOINTS = torch.tensor(
    [0.0, 0.0, 0.0, -1.57, 1.57, 0.0],  # 예시: home pose
    dtype=torch.float32,
)


# -------------------------------
# Rewards
# -------------------------------
def joint_target_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 joint error"""
    # 로봇 조인트 상태 직접 접근
    joint_pos = env.scene["robot"].data.joint_pos  # shape: [num_envs, num_joints]
    target = TARGET_JOINTS.to(env.device)
    target = target.unsqueeze(0).expand_as(joint_pos)
    return torch.mean((joint_pos - target) ** 2, dim=-1)


def joint_target_tanh(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Smoothed reward"""
    joint_pos = env.scene["robot"].data.joint_pos
    target = TARGET_JOINTS.to(env.device)
    target = target.unsqueeze(0).expand_as(joint_pos)
    error = torch.mean((joint_pos - target) ** 2, dim=-1)
    return 1.0 - torch.tanh(error)
