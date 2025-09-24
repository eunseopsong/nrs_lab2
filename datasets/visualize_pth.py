# SPDX-License-Identifier: BSD-3-Clause
"""
visualize_pth.py
- .pth 파일 내부 파라미터 확인 및 joint target 값 시각화
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ------------------------------------------------------
# 1. 모델 정의 (V17 / ResiP 기반 MLP 정책 예시)
#    - 반드시 실제 학습할 때 사용한 구조와 동일해야 함
# ------------------------------------------------------
class PolicyMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------
# 2. .pth 파일 로드
# ------------------------------------------------------
def load_model(pth_file, input_dim=6, hidden_dim=128, output_dim=6):
    ckpt = torch.load(pth_file, map_location="cpu")

    # state_dict 추출
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # 모델 생성 및 weight 로드
    model = PolicyMLP(input_dim, hidden_dim, output_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ------------------------------------------------------
# 3. 시각화
# ------------------------------------------------------
def visualize_policy(model, num_samples=100, input_dim=6):
    # 랜덤 상태 입력 (num_samples x input_dim)
    xs = torch.randn(num_samples, input_dim)
    ys = model(xs).detach().numpy()

    # joint 별 시각화
    plt.figure(figsize=(10, 6))
    for j in range(ys.shape[1]):
        plt.plot(ys[:, j], label=f"Joint {j+1}")
    plt.legend()
    plt.title("Predicted Joint Targets from .pth")
    plt.xlabel("Sample index")
    plt.ylabel("Joint target value")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------
# 4. 메인 실행
# ------------------------------------------------------
if __name__ == "__main__":
    pth_file = "bc_policy.pth"  # 🔥 분석할 .pth 파일 경로 지정
    model = load_model(pth_file, input_dim=6, hidden_dim=128, output_dim=6)
    visualize_policy(model, num_samples=200, input_dim=6)

