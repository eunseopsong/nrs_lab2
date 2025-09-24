# ~/nrs_lab2/scripts/behavior_cloning/bc_play.py
import torch
import torch.nn as nn
import gymnasium as gym
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# === Policy 정의 (학습 때랑 동일해야 함) ===
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)

# === 환경 생성 ===
env = gym.make("Nrs-UR10e-Manager-v1")
env = SkrlVecEnvWrapper(env, ml_framework="torch")

obs, _ = env.reset()

# === Policy 로드 ===
policy = BCPolicy(obs_dim=6, act_dim=6)
policy.load_state_dict(torch.load("logs/bc_policy.pth"))
policy.eval()

# === 실행 ===
for step in range(300):   # 원하는 step 수만큼 실행
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = policy(obs_tensor).cpu().numpy()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated.any() or truncated.any():
        obs, _ = env.reset()

env.close()

