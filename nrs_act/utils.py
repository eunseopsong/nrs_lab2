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
    """
    HDF5 구조 가정:

      - /observations/qpos              : (T, 6)
      - /observations/qvel              : (T, 6)   # 지금은 안 쓰지만 그대로 둠
      - /observations/images/cam_front  : (T, H, W, 3)
      - /observations/images/cam_head   : (T, H, W, 3)
      - /action/joints                  : (T, 6)
      - /action/ft                      : (T, 3)   # 없으면 0으로 채움
        * 혹은 /action                  : (T, 6) 또는 (T, 9) 구버전 호환용

    이 Dataset 은 한 시점의 (이미지 + qpos)를 observation 으로 꺼내고,
    그 시점 이후의 action 시퀀스(길이 T, dim=9)를 패딩해서 반환한다.
    """

    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super().__init__()
        self.episode_ids  = list(episode_ids)
        self.dataset_dir  = dataset_dir
        self.camera_names = camera_names
        self.norm_stats   = norm_stats
        self.is_sim       = False  # 기본값

        # 에피소드가 하나도 없으면 여기서 끝
        if len(self.episode_ids) > 0:
            # 맨 처음 에피소드 한 번만 열어서 is_sim 값 알아낸다
            _id0  = self.episode_ids[0]
            path0 = os.path.join(self.dataset_dir, f"episode_{_id0}.hdf5")
            with h5py.File(path0, "r") as root:
                self.is_sim = bool(root.attrs.get("sim", False))

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # 원래 코드 구조 유지: 한 시점의 observation + 그 이후 전체 action 시퀀스
        sample_full_episode = False

        episode_id   = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            is_sim = bool(root.attrs.get("sim", False))

            # -------------------------
            # 1) action(joints + ft) 읽기
            #    joints: (T, 6), ft: (T, 3) → concat → (T, 9)
            #    * /action/joints + /action/ft 조합이 우선
            #    * 그 다음 /action (T,9) 또는 (T,6) fallback
            # -------------------------
            if "/action/joints" in root:
                joints_ds = root["/action/joints"]              # (T, 6)
                T = joints_ds.shape[0]

                if "/action/ft" in root:
                    ft_ds = root["/action/ft"]                  # (T, 3)
                    assert ft_ds.shape[0] == T, \
                        f"ft length mismatch: joints {T}, ft {ft_ds.shape[0]}"
                    ft = ft_ds[()]                              # (T, 3)
                else:
                    ft = np.zeros((T, 3), dtype=np.float32)     # ft 없으면 0
                joints = joints_ds[()]                          # (T, 6)
                action_full = np.concatenate([joints, ft], axis=-1)  # (T, 9)

            elif "action" in root:
                action_raw = root["action"][()]                 # (T, D)
                T, D = action_raw.shape
                if D == 9:
                    # 이미 joints+ft 로 합쳐진 형태
                    action_full = action_raw
                elif D == 6:
                    # joints만 있는 경우 → ft=0 으로 채워서 (T,9) 맞춤
                    ft = np.zeros((T, 3), dtype=np.float32)
                    action_full = np.concatenate([action_raw, ft], axis=-1)
                else:
                    raise ValueError(
                        f"[EpisodicDataset] Unsupported action dim {D} in {dataset_path}"
                    )
            else:
                raise KeyError(f"[EpisodicDataset] No action dataset in {dataset_path}")

            original_action_shape = action_full.shape          # (T, 9)
            episode_len = original_action_shape[0]

            # -------------------------
            # 2) 시작 시점 선택
            # -------------------------
            if sample_full_episode:
                start_ts = 0
            else:
                # 0 ~ T-1 중 하나
                start_ts = np.random.randint(0, episode_len)

            # -------------------------
            # 3) 관측 한 프레임 (qpos + 이미지)
            # -------------------------
            qpos = root["/observations/qpos"][start_ts]  # (6,)
            qvel = root["/observations/qvel"][start_ts]  # 지금은 사용 안 함

            image_dict = {}
            for cam_name in self.camera_names:
                # 예: /observations/images/cam_front
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            # -------------------------
            # 4) action 시퀀스 (joints+ft, dim=9)
            #    - sim이면 start_ts 이후
            #    - real이면 기존 코드대로 (start_ts - 1)부터
            # -------------------------
            if is_sim:
                action = action_full[start_ts:]             # (T - start_ts, 9)
                action_len = episode_len - start_ts
            else:
                start_for_action = max(0, start_ts - 1)
                action = action_full[start_for_action:]     # (T - start_for_action, 9)
                action_len = episode_len - start_for_action

        # dataset 멤버에도 저장 (load_data에서 쓰려고)
        self.is_sim = is_sim

        # -------------------------
        # 5) 패딩해서 길이 맞추기
        # -------------------------
        padded_action = np.zeros(original_action_shape, dtype=np.float32)  # (T, 9)
        padded_action[:action_len] = action

        is_pad = np.zeros(episode_len, dtype=np.float32)
        is_pad[action_len:] = 1.0

        # -------------------------
        # 6) 여러 카메라 한 텐서로 쌓기 -> (K, H, W, C)
        # -------------------------
        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)   # (K, H, W, C)

        # -------------------------
        # 7) torch tensor로 변환
        # -------------------------
        image_data  = torch.from_numpy(all_cam_images)         # (K, H, W, C)
        qpos_data   = torch.from_numpy(qpos).float()           # (6,)
        action_data = torch.from_numpy(padded_action).float()  # (T, 9)
        is_pad      = torch.from_numpy(is_pad).bool()          # (T,)

        # 채널 순서 바꾸기: (K, H, W, C) -> (K, C, H, W)
        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0   # [0,1] 정규화

        # -------------------------
        # 8) normalization (action: dim=9, qpos: dim=6)
        # -------------------------
        action_mean = torch.tensor(self.norm_stats["action_mean"]).float()  # (9,)
        action_std  = torch.tensor(self.norm_stats["action_std"]).float()   # (9,)
        qpos_mean   = torch.tensor(self.norm_stats["qpos_mean"]).float()    # (6,)
        qpos_std    = torch.tensor(self.norm_stats["qpos_std"]).float()     # (6,)

        action_data = (action_data - action_mean) / action_std   # (T, 9)
        qpos_data   = (qpos_data - qpos_mean) / qpos_std         # (6,)

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, episode_ids):
    """
    HDF5 구조:
      - /observations/qpos      : (T, 6)
      - /action/joints          : (T, 6)
      - /action/ft              : (T, 3)
        * 또는 /action          : (T, 6) 또는 (T, 9)

    여기서는 "실제 network가 예측해야 하는 action 벡터"와 동일한
    차원(D=9)에 대해 mean/std 를 계산한다.
    (joints(6) + ft(3) → dim=9)
    """
    all_qpos_data   = []
    all_action_data = []

    for episode_idx in episode_ids:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        if not os.path.exists(dataset_path):
            print(f"[WARN] get_norm_stats: {dataset_path} not found, skip")
            continue

        with h5py.File(dataset_path, "r") as root:
            # qpos: (T, 6)
            if "/observations/qpos" in root:
                qpos = root["/observations/qpos"][()]   # (T, 6)
            else:
                qpos = root["observations"]["qpos"][()]

            # --------- action_full (T, 9) 구성 ---------
            if "/action/joints" in root:
                joints = root["/action/joints"][()]            # (T, 6)
                T = joints.shape[0]
                if "/action/ft" in root:
                    ft = root["/action/ft"][()]                # (T, 3)
                    assert ft.shape[0] == T, \
                        f"[get_norm_stats] ft length mismatch in {dataset_path}"
                else:
                    ft = np.zeros((T, 3), dtype=np.float32)
                action_full = np.concatenate([joints, ft], axis=-1)  # (T, 9)

            elif "action" in root:
                action_raw = root["action"][()]                # (T, D)
                T, D = action_raw.shape
                if D == 9:
                    action_full = action_raw
                elif D == 6:
                    ft = np.zeros((T, 3), dtype=np.float32)
                    action_full = np.concatenate([action_raw, ft], axis=-1)
                else:
                    print(
                        f"[WARN] get_norm_stats: Unsupported action dim {D} "
                        f"in {dataset_path}, skip"
                    )
                    continue
            else:
                print(f"[WARN] get_norm_stats: no action in {dataset_path}, skip")
                continue

        all_qpos_data.append(torch.from_numpy(qpos))          # (T, 6)
        all_action_data.append(torch.from_numpy(action_full)) # (T, 9)

    if len(all_qpos_data) == 0:
        raise RuntimeError(f"[get_norm_stats] No valid episodes found in {dataset_dir}")

    # (N, T, D) 로 쌓기
    all_qpos_data   = torch.stack(all_qpos_data)       # (N, T, 6)
    all_action_data = torch.stack(all_action_data)     # (N, T, 9)

    # action (joints+ft, dim=9)
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)  # (1,1,9)
    action_std  = all_action_data.std(dim=[0, 1], keepdim=True)   # (1,1,9)
    action_std  = torch.clip(action_std, 1e-2, np.inf)

    # qpos
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)      # (1,1,6)
    qpos_std  = all_qpos_data.std(dim=[0, 1], keepdim=True)       # (1,1,6)
    qpos_std  = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(),  # (9,)
        "action_std":  action_std.numpy().squeeze(),   # (9,)
        "qpos_mean":   qpos_mean.numpy().squeeze(),    # (6,)
        "qpos_std":    qpos_std.numpy().squeeze(),     # (6,)
        "example_qpos": all_qpos_data[0].numpy(),      # 첫 에피소드 예시
    }
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    # num_episodes는 무시하고 실제 있는 파일만 쓴다
    episode_ids = _list_episode_ids(dataset_dir)
    if len(episode_ids) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {dataset_dir}")

    print(f"\nData from: {dataset_dir}")
    print(f"Found {len(episode_ids)} episodes: {episode_ids}\n")

    # train/val split
    train_ratio = 0.8
    shuffled = np.random.permutation(episode_ids)
    split_idx = int(len(shuffled) * train_ratio)
    train_ids = shuffled[:split_idx]
    val_ids   = shuffled[split_idx:]

    # 정규화 통계는 전체 episode 기준
    norm_stats = get_norm_stats(dataset_dir, episode_ids)

    # dataset & dataloader
    train_dataset = EpisodicDataset(train_ids, dataset_dir, camera_names, norm_stats)
    val_dataset   = EpisodicDataset(val_ids,   dataset_dir, camera_names, norm_stats)

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
