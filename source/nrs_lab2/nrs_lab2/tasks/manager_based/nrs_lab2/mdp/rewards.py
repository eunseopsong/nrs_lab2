# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -------------------
# Globals
# -------------------
_bc_policy = None
_prev_error = None   # convergence 보상 계산용


# -------------------
# Policy load
# -------------------
def load_bc_policy(env: ManagerBasedRLEnv, env_ids, file_path: str):
    """BC policy(.pth) 로드 (reset 시 1회 호출)"""
    global _bc_policy, _prev_error
    state_dict = torch.load(file_path, map_location="cpu")

    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_dim=6, hidden=128, out_dim=6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, out_dim)
            )
        def forward(self, x): return self.net(x)

    model = MLP()
    model.load_state_dict(state_dict)   # ✅ 구조 동일하니 로딩 성공
    model.eval()
    _bc_policy = model.to(env.device)
    _prev_error = None
    print(f"[BC] Loaded policy from {file_path} with architecture: 6 -> 128 -> 128 -> 6")



def get_bc_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    global _bc_policy
    if _bc_policy is None:
        raise RuntimeError("BC policy not loaded. Did you register load_bc_policy?")
    obs = env.scene["robot"].data.joint_pos   # ✅ joint_pos 6차원만 입력
    with torch.no_grad():
        action = _bc_policy(obs)
    return action


# -------------------
# Reward functions
# -------------------
def joint_target_error_weighted(env: ManagerBasedRLEnv, weights=None) -> torch.Tensor:
    """Joint-wise weighted tracking error vs BC policy action"""
    if weights is None:
        weights = [2.0, 2.0, 2.0, 5.0, 1.0, 1.0]

    q = env.scene["robot"].data.joint_pos
    target = get_bc_action(env)  # BC policy가 생성한 목표 joint pos
    diff = q - target
    weighted_error = torch.sum(torch.tensor(weights, device=env.device) * (diff ** 2), dim=-1)

    if env.common_step_counter % 10 == 0:
        current_time = env.common_step_counter * env.step_dt
        print(f"[Step {env.common_step_counter} | Time {current_time:.2f}s] "
              f"Target[0]: {target[0].detach().cpu().numpy()} "
              f"Current[0]: {q[0].detach().cpu().numpy()} "
              f"WeightedError[0]: {weighted_error[0].item():.6f}")

    return -weighted_error


def fast_convergence_reward(env: ManagerBasedRLEnv, weights=None) -> torch.Tensor:
    """
    에러 감소 속도 기반 보상
    - 이전 error 대비 감소하면 보상 부여
    """
    global _prev_error
    if weights is None:
        weights = [2.0, 2.0, 2.0, 5.0, 1.0, 1.0]

    q = env.scene["robot"].data.joint_pos
    target = get_bc_action(env)
    diff = q - target
    weighted_error = torch.sum(torch.tensor(weights, device=env.device) * (diff ** 2), dim=-1)

    if _prev_error is None:
        _prev_error = weighted_error.clone()
        return torch.zeros_like(weighted_error)

    delta = _prev_error - weighted_error
    _prev_error = weighted_error.clone()
    reward = torch.clamp(delta, min=0.0)

    if env.common_step_counter % 100 == 0:
        print(f"[Step {env.common_step_counter}] FastConvergenceReward[0]: {reward[0].item():.6f}")

    return reward


# -------------------
# Termination function
# -------------------
def reached_end(env: ManagerBasedRLEnv) -> torch.Tensor:
    # BC 기반 imitation은 step 제한까지만 진행
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
