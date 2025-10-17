import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ==============================
# 1️⃣ Dataset Loader
# ==============================
class H5JointDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        with h5py.File(file_path, "r") as f:
            # 기본 키 구조 예시: ["joint_positions", "joint_velocities"]
            if "joint_positions" in f:
                self.joint_pos = np.array(f["joint_positions"])
            else:
                raise KeyError("Dataset must contain 'joint_positions' dataset")

        # 시퀀스 길이 설정 (LSTM 입력 길이)
        self.seq_len = 10  # 최근 10개 상태로 다음 상태 예측
        self.input_dim = self.joint_pos.shape[1]

    def __len__(self):
        return len(self.joint_pos) - self.seq_len

    def __getitem__(self, idx):
        x = self.joint_pos[idx : idx + self.seq_len]
        y = self.joint_pos[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ==============================
# 2️⃣ LSTM Policy Model
# ==============================
class LSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 마지막 타임스텝만 사용


# ==============================
# 3️⃣ 학습 루프
# ==============================
def train_bc_model(
    file_path="/home/eunseop/nrs_lab2/datasets/joint_recording.h5",
    save_path="/home/eunseop/nrs_lab2/datasets/model_bc.pt",
    epochs=100,
    batch_size=64,
    lr=1e-3,
):
    # 데이터셋 로드
    dataset = H5JointDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    input_dim = dataset.input_dim
    model = LSTMPolicy(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=input_dim)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X, Y in dataloader:
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1:03d}] Loss = {epoch_loss / len(dataloader):.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ LSTM BC model saved to {save_path}")


# ==============================
# 4️⃣ 실행 엔트리포인트
# ==============================
if __name__ == "__main__":
    train_bc_model()
