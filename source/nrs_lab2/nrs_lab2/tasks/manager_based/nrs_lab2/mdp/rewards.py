# SPDX-License-Identifier: BSD-3-Clause
"""
Rewards & BC utilities
"""

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_bc_policy = None

def load_bc_policy(env: "ManagerBasedRLEnv", env_ids, file_path: str):
    """BC policy 불러오기"""
    global _bc_policy
    _bc_policy = torch.load(file_path, map_location=env.device)
    _bc_policy.eval()
    print(f"[INFO] BC policy loaded from {file_path}")

def get_bc_action(env: "ManagerBasedRLEnv", obs: torch.Tensor) -> torch.Tensor:
    """obs → predicted joint pos"""
    global _bc_policy
    if _bc_policy is None:
        raise RuntimeError("BC policy not loaded.")
    with torch.no_grad():
        q_pred = _bc_policy(obs[:, :6])  # joint_pos 부분만 사용
    return q_pred

# Dummy reward/termination (학습 안 함)
def dummy_reward(env: "ManagerBasedRLEnv"):
    return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

def dummy_termination(env: "ManagerBasedRLEnv"):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
