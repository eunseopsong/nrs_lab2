# SPDX-License-Identifier: BSD-3-Clause
"""
Behavior Cloning (BC) training script for UR10e joint trajectory tracking.
- Loads dataset from HDF5 file
- Trains a simple MLP policy (obs -> action)
- Uses supervised MSE loss
"""

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset loader
# -----------------------------
class JointBCDataset(Dataset):
    def __init__(self, hdf5_path: str, dataset_key: str = "joint_positions", obs_type="delta"):
        with h5py.File(hdf5_path, "r") as f:
            data = f[dataset_key][:]   # shape: [T, DoF]
        self.q_targets = torch.tensor(data, dtype=torch.float32)

        # 관측 정의: (q_current, q_target) -> action (q_target or delta)
        # 여기서는 간단히 delta(q) = q_target - q_current
        self.obs_type = obs_type

    def __len__(self):
        return len(self.q_targets) - 1

    def __getitem__(self, idx):
        q_current = self.q_targets[idx]
        q_target = self.q_targets[idx + 1]   # 다음 step을 target으로 둠

        if self.obs_type == "delta":
            obs = q_current
            action = q_target - q_current
        elif self.obs_type == "absolute":
            obs = q_current
            action = q_target
        else:
            raise ValueError("Unknown obs_type")

        return obs, action


# -----------------------------
# Policy Network (MLP)
# -----------------------------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Training loop
# -----------------------------
def train_bc(hdf5_path: str, num_epochs=50, batch_size=64, lr=1e-3, obs_type="delta"):
    # Dataset
    dataset = JointBCDataset(hdf5_path, "joint_positions", obs_type=obs_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = dataset[0][0].shape[0]
    act_dim = dataset[0][1].shape[0]

    # Model + optimizer
    policy = MLPPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for obs, action in dataloader:
            pred = policy(obs)
            loss = criterion(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    # Save model
    torch.save(policy.state_dict(), "bc_policy.pth")
    print("✅ Saved BC policy to bc_policy.pth")

    return policy


if __name__ == "__main__":
    train_bc(
        hdf5_path="/home/eunseop/nrs_lab2/datasets/joint_recording.h5",
        num_epochs=50,
        batch_size=64,
        lr=1e-3,
        obs_type="delta"   # "absolute" 로도 가능
    )

