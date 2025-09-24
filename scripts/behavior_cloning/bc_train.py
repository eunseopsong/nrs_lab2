# bc_train.py (dataset-only BC, IsaacSim 불필요)

import argparse
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="/home/eunseop/nrs_lab2/datasets/joint_recording.h5")
parser.add_argument("--dataset_key", type=str, default="joint_positions")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--save_path", type=str, default="logs/bc_policy.pth")
args = parser.parse_args()

# -------------------
# Load dataset
# -------------------
with h5py.File(args.dataset, "r") as f:
    data = f[args.dataset_key][:]
print(f"[INFO] Loaded dataset {args.dataset} with shape {data.shape}")
dataset = torch.tensor(data, dtype=torch.float32)

obs_dim = dataset.shape[1]
act_dim = dataset.shape[1]   # 여기서는 obs=joint_pos, action=joint_pos (identity imitation)

# -------------------
# Define Policy
# -------------------
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

policy = BCPolicy(obs_dim, act_dim).to(args.device)
optimizer = optim.Adam(policy.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

# -------------------
# Training Loop
# -------------------
n_samples = dataset.shape[0]
indices = np.arange(n_samples)

for epoch in range(args.epochs):
    np.random.shuffle(indices)
    total_loss = 0.0

    for start in range(0, n_samples, args.batch_size):
        end = start + args.batch_size
        batch_idx = indices[start:end]

        obs_batch = dataset[batch_idx, :]
        target_batch = dataset[batch_idx, :]

        pred = policy(obs_batch.to(args.device))
        loss = loss_fn(pred, target_batch.to(args.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss:.6f}")

# -------------------
# Save Policy
# -------------------
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
torch.save(policy.state_dict(), args.save_path)
print(f"[INFO] Saved BC policy to {args.save_path}")

