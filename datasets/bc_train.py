import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bc_dataset import BCDataset


# ================================
# 1. Policy 네트워크 정의 (MLP)
# ================================
class Policy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# 2. 학습 루프
# ================================
def train_bc(h5_path="bc_dataset.h5", num_epochs=50, batch_size=32, lr=1e-3):
    # 데이터셋 로드
    dataset = BCDataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    model = Policy(obs_dim=6, act_dim=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 학습
    for epoch in range(num_epochs):
        total_loss = 0.0
        for obs, act in dataloader:
            pred = model(obs)
            loss = loss_fn(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    # 모델 저장
    torch.save(model.state_dict(), "bc_policy.pth")
    print("✅ Saved trained policy → bc_policy.pth")


# ================================
# 3. 실행 엔트리포인트
# ================================
if __name__ == "__main__":
    train_bc()

