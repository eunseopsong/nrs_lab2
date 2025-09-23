# scripts/behavior_cloning/bc_play.py
import torch
from isaaclab.envs import ManagerBasedRLEnv
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg

# 1. 환경 불러오기
env_cfg = UR10eSpindleEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)

# 2. BC policy 불러오기
policy = torch.load("bc_policy.pth")

# 3. 실행 루프
obs, _ = env.reset()
for step in range(1000):
    with torch.no_grad():
        action = policy(obs)  # 네트워크 forward
    obs, reward, done, info = env.step(action)
    env.render()

