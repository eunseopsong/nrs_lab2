import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class BCDataset(Dataset):
    def __init__(self, h5_path: str):
        self.file = h5py.File(h5_path, "r")
        self.obs = self.file["observations"][:]  # (N, 6)
        self.acts = self.file["actions"][:]      # (N, 6)

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, idx):
        obs = torch.tensor(self.obs[idx], dtype=torch.float32)
        act = torch.tensor(self.acts[idx], dtype=torch.float32)
        return obs, act

if __name__ == "__main__":
    # 데이터셋 불러오기
    dataset = BCDataset("bc_dataset.h5")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Dataset size:", len(dataset))

    # 샘플 확인
    for obs, act in dataloader:
        print("obs batch shape:", obs.shape)
        print("act batch shape:", act.shape)
        break

