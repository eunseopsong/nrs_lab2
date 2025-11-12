import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader


# -----------------------------------------------------------
# 유틸: dataset_dir 안에 실제 있는 episode_* 만 찾아서 id 리스트로 돌려준다.
# -----------------------------------------------------------
def _list_episode_ids(dataset_dir: str):
    ids = []
    if not os.path.isdir(dataset_dir):
        return ids

    for fname in os.listdir(dataset_dir):
        if not fname.startswith("episode_") or not fname.endswith(".hdf5"):
            continue
        # episode_12.hdf5 -> 12
        stem = fname[len("episode_") : -len(".hdf5")]
        if stem.isdigit():
            ids.append(int(stem))
    ids.sort()
    return ids


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super().__init__()
        self.episode_ids = list(episode_ids)
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = False  # 기본값

        # 에피소드가 하나도 없으면 여기서 끝
        if len(self.episode_ids) > 0:
            # 맨 처음 에피소드 한 번만 열어서 is_sim 값 알아낸다
            _id0 = self.episode_ids[0]
            path0 = os.path.join(self.dataset_dir, f"episode_{_id0}.hdf5")
            with h5py.File(path0, "r") as root:
                self.is_sim = bool(root.attrs.get("sim", False))

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # 그냥 원본 유지

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = bool(root.attrs.get("sim", False))

            original_action_shape = root["/action"].shape  # (T, 6) or (T, 12)
            episode_len = original_action_shape[0]

            # 1) 시작 시점 뽑기
            if sample_full_episode:
                start_ts = 0
            else:
                # 0 ~ T-1 중 하나
                start_ts = np.random.randint(0, episode_len)

            # 2) 관측 한 프레임만
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]  # 지금은 안 쓰지만 남겨둠

            image_dict = {}
            for cam_name in self.camera_names:
                # 예: /observations/images/cam_front
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            # 3) 액션은 start_ts 이후 전부
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                # 원래 코드가 이렇게 되어 있어서 유지
                action = root["/action"][max(0, start_ts - 1) :]
                action_len = episode_len - max(0, start_ts - 1)

        # dataset 멤버에도 저장 (load_data에서 쓰려고)
        self.is_sim = is_sim

        # 4) 패딩해서 길이 맞추기
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action

        is_pad = np.zeros(episode_len, dtype=np.float32)
        is_pad[action_len:] = 1.0

        # 5) 여러 카메라 쌓기 -> (K, H, W, C)
        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)

        # 6) torch tensor로 변환
        image_data = torch.from_numpy(all_cam_images)            # (K, H, W, C)
        qpos_data = torch.from_numpy(qpos).float()               # (D,)
        action_data = torch.from_numpy(padded_action).float()    # (T, D)
        is_pad = torch.from_numpy(is_pad).bool()                 # (T,)

        # 채널 순서 바꾸기: (K, H, W, C) -> (K, C, H, W)
        image_data = torch.einsum("k h w c -> k c h w", image_data)
        # 정규화
        image_data = image_data / 255.0

        # ---- 여기서부터 normalization ----
        action_mean = torch.tensor(self.norm_stats["action_mean"]).float()
        action_std = torch.tensor(self.norm_stats["action_std"]).float()
        qpos_mean = torch.tensor(self.norm_stats["qpos_mean"]).float()
        qpos_std = torch.tensor(self.norm_stats["qpos_std"]).float()

        action_data = (action_data - action_mean) / action_std
        qpos_data = (qpos_data - qpos_mean) / qpos_std

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, episode_ids):
    """실제로 있는 episode들만 모아서 평균/표준편차 계산."""
    all_qpos_data = []
    all_action_data = []

    for episode_idx in episode_ids:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]   # (T, D)
            action = root["/action"][()]            # (T, D)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # (N, T, D) 로 쌓기
    all_qpos_data = torch.stack(all_qpos_data)       # (N, T, D)
    all_action_data = torch.stack(all_action_data)   # (N, T, D)

    # action
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # qpos
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,  # 마지막 에피소드 qpos 하나 넣어둠
    }
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    # num_episodes는 무시하고 실제 있는 파일만 쓴다
    episode_ids = _list_episode_ids(dataset_dir)
    if len(episode_ids) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {dataset_dir}")

    print(f"\nData from: {dataset_dir}")
    print(f"Found {len(episode_ids)} episodes: {episode_ids}\n")

    # split
    train_ratio = 0.8
    shuffled = np.random.permutation(episode_ids)
    split_idx = int(len(shuffled) * train_ratio)
    train_ids = shuffled[:split_idx]
    val_ids = shuffled[split_idx:]

    # stats는 전체 episode 기준으로
    norm_stats = get_norm_stats(dataset_dir, episode_ids)

    # dataset & dataloader
    train_dataset = EpisodicDataset(train_ids, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_ids, dataset_dir, camera_names, norm_stats)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# ================== 아래는 원래 있던 유틸들 그대로 ==================

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------
# 모델 체크포인트 로드 유틸
# -----------------------------------------------------------
def load_checkpoint(model, ckpt_path, device="cuda"):
    """학습된 모델 파라미터(.ckpt) 파일을 로드합니다."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[load_checkpoint] checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # 일반적인 PyTorch state_dict 구조
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError(f"[load_checkpoint] Unexpected checkpoint format: {type(checkpoint)}")

    print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    return model
