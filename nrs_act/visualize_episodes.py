import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt

# 네 환경용 상수
from custom_constants import DT

# UR10e 6축 기준
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def load_hdf5(dataset_dir, dataset_name):
    """
    dataset_dir/episode_X.hdf5 형태로 저장된 ACT-style 파일을 읽는다.
    """
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        is_sim = bool(root.attrs.get("sim", False))
        qpos = root["/observations/qpos"][()]      # (T, D)
        qvel = root["/observations/qvel"][()]      # (T, D)
        action = root["/action"][()]               # (T, D)

        # 카메라 이름 자동 수집
        image_dict = {}
        img_group = root["/observations/images/"]
        for cam_name in img_group.keys():
            image_dict[cam_name] = img_group[cam_name][()]  # (T, H, W, C)

    return qpos, qvel, action, image_dict


def main(args):
    dataset_dir = args["dataset_dir"]
    episode_idx = args.get("episode_idx", None)
    if episode_idx is None:
        episode_idx = 0

    dataset_name = f"episode_{episode_idx}"

    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)

    # 비디오 저장
    video_path = os.path.join(dataset_dir, dataset_name + "_video.mp4")
    save_videos(image_dict, DT, video_path=video_path)

    # 조인트 플롯
    plot_path = os.path.join(dataset_dir, dataset_name + "_qpos.png")
    visualize_joints(qpos, action, plot_path=plot_path)


def save_videos(video, dt, video_path=None):
    """
    video: dict(cam_name -> (T, H, W, C))
    두 카메라를 옆으로 붙여서 하나의 mp4로 저장
    """
    if not isinstance(video, dict):
        raise ValueError("Expected a dict of {camera_name: np.ndarray}")

    # 카메라 이름 정렬: 우리가 주로 쓰는 이름이 있으면 그 순서로
    cam_names = list(video.keys())
    # 우선순위 정해서 정렬
    priority = {"cam_front": 0, "cam_head": 1}
    cam_names.sort(key=lambda x: priority.get(x, 99))

    # 첫 카메라로 기본 shape 가져옴
    first_cam = cam_names[0]
    frames = video[first_cam]  # (T, H, W, C)
    T, H, W, C = frames.shape

    # 전체 너비 = 카메라 개수 * W
    total_w = W * len(cam_names)
    fps = int(1 / dt)

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (total_w, H),
    )

    for t in range(T):
        row_imgs = []
        for cam in cam_names:
            img = video[cam][t]  # (H, W, C), RGB
            img = img[:, :, ::-1]  # RGB -> BGR
            row_imgs.append(img)
        row = np.concatenate(row_imgs, axis=1)
        out.write(row)

    out.release()
    print(f"Saved video to: {video_path}")


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    """
    qpos_list: (T, D)
    command_list: (T, D)
    D가 6일 때 6개만 그린다.
    """
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = "State", "Command"

    qpos = np.array(qpos_list)      # (T, D)
    command = np.array(command_list)
    T, D = qpos.shape

    fig, axs = plt.subplots(D, 1, figsize=(8, 2.5 * D))

    if D == 1:
        axs = [axs]

    for dim_idx in range(D):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.plot(command[:, dim_idx], label=label2, alpha=0.7)
        # 조인트 이름이 모자라면 인덱스로 표시
        joint_name = JOINT_NAMES[dim_idx] if dim_idx < len(JOINT_NAMES) else f"joint_{dim_idx}"
        ax.set_title(f"Joint {dim_idx}: {joint_name}")
        ax.legend()
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
        print(f"Saved qpos plot to: {plot_path}")
        plt.close()
    else:
        plt.show()


def visualize_timestamp(t_list, dataset_path):
    """
    이건 원본에 있던 함수 그대로 두되, 지금은 안 써도 됨.
    """
    plot_path = dataset_path.replace(".pkl", "_timestamp.png")
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h * 2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 1e-9)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title("Camera frame timestamps")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    ax = axs[1]
    ax.plot(np.arange(len(t_float) - 1), np.diff(t_float))
    ax.set_title("dt")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved timestamp plot to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing episode_*.hdf5")
    parser.add_argument("--episode_idx", type=int, required=False, help="Episode index (default: 0)")
    main(vars(parser.parse_args()))
