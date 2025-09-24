# SPDX-License-Identifier: BSD-3-Clause
"""
Minimal Behavior Cloning (BC) Play Script for UR10e + Spindle
- Loads trained bc_policy.pth
- Runs in Manager-based environment (PolishingPoseHoldEnvCfg_PLAY)
- No reward, no training. Just inference & rendering.
"""

import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.polishing_env_cfg import PolishingPoseHoldEnvCfg_PLAY


# -----------------------------
# 1. Minimal MLP Policy
# -----------------------------
class BCPolicy(nn.Module):
    def __init__(self, obs_dim=18, act_dim=6, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),   # actions scaled to [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# -----------------------------
# 2. Main Loop
# -----------------------------
def main():
    # 환경 불러오기
    env_cfg = PolishingPoseHoldEnvCfg_PLAY()
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 정책 로드
    policy = BCPolicy(obs_dim=18, act_dim=6)
    policy.load_state_dict(torch.load("datasets/bc_policy.pth"))
    policy.eval()

    # 환경 초기화
    obs, _ = env.reset()

    # Play Loop
    while True:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = policy(obs_tensor).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        # 한 개라도 종료되면 reset
        if terminated.any() or truncated.any():
            obs, _ = env.reset()

        env.render()


if __name__ == "__main__":
    main()
