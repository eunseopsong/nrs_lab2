import torch
from bc_train import Policy
from isaaclab.envs import ManagerBasedRLEnv
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import PolishingPoseHoldEnvCfg


def main():
    # ================================
    # 1. 학습된 Policy 불러오기
    # ================================
    model = Policy(obs_dim=6, act_dim=6)
    model.load_state_dict(torch.load("bc_policy.pth"))
    model.eval()
    print("✅ Loaded trained policy: bc_policy.pth")

    # ================================
    # 2. Isaac Lab 환경 초기화
    # ================================
    env = ManagerBasedRLEnv(cfg=PolishingPoseHoldEnvCfg())
    obs, _ = env.reset()
    done = False
    step_count = 0

    # ================================
    # 3. Policy 실행 루프
    # ================================
    while not done and step_count < 500:  # 최대 500 step만 실행
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = model(obs_tensor).numpy()

        obs, reward, done, info = env.step(action)
        step_count += 1

    print("✅ Finished BC rollout.")


if __name__ == "__main__":
    main()

