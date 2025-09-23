# SPDX-License-Identifier: BSD-3-Clause
"""
Run Behavior Cloning (BC) policy in Isaac Lab
"""

import torch
import numpy as np
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg

# 1. 환경 불러오기
env_cfg = UR10eSpindleEnvCfg()
env = DirectRLEnv(cfg=env_cfg)

# 2. 학습된 BC policy 로드
policy = torch.load("bc_policy.pth", map_location=env.device)
policy.eval()

# 3. rollout 실행
obs, _ = env.reset()
done = False
step = 0

while not done:
    # policy는 joint position 예측 (obs → action)
    with torch.no_grad():
        action = policy(torch.tensor(obs, dtype=torch.float32, device=env.device))

    # 환경 step
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 50 == 0:
        print(f"[Step {step}] Reward: {reward.mean().item():.6f}")

    done = bool(terminated.any() or truncated.any())

