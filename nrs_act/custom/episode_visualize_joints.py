#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하나의 episode_*.hdf5 파일에서 joint(qpos) 궤적 시각화.
"""

import os
import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 기본 경로 (필요하면 수정)
# -------------------------------------------------
EPISODE_DIR = "/home/eunseop/nrs_lab2/datasets/ACT/1114_1643/episodes"


def load_qpos_from_episode(ep_dir: str, episode_idx: int) -> np.ndarray:
    """
    ep_dir/episode_<idx>.hdf5 에서 /observations/qpos dataset을 읽어서 (T, DoF) 반환
    """
    fname = f"episode_{episode_idx}.hdf5"
    path = os.path.join(ep_dir, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Episode file not found: {path}")

    with h5py.File(path, "r") as f:
        # UR10e용 demo는 /observations/qpos 에 joint가 들어있다고 가정
        if "/observations/qpos" not in f:
            raise KeyError("'/observations/qpos' dataset not found in file.")
        qpos = f["/observations/qpos"][()]   # (T, DoF)

    return qpos


def plot_joints(qpos: np.ndarray, title_prefix: str = ""):
    """
    qpos : (T, DoF)
    각 joint별로 subplot에 step vs rad 플롯
    """
    T, dof = qpos.shape
    t = np.arange(T)

    joint_names = [f"j{i}" for i in range(dof)]

    n_rows = int(np.ceil(dof / 2))
    n_cols = 2 if dof > 1 else 1

    plt.figure(figsize=(12, 3.5 * n_rows))

    for j in range(dof):
        plt.subplot(n_rows, n_cols, j + 1)
        plt.plot(t, qpos[:, j], linewidth=1.5)
        plt.xlabel("step")
        plt.ylabel("rad")
        plt.title(f"{title_prefix}{joint_names[j]}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ep_idx",
        type=int,
        default=0,
        help="시각화할 episode 인덱스 (예: 0 -> episode_0.hdf5)",
    )
    parser.add_argument(
        "--ep_dir",
        type=str,
        default=EPISODE_DIR,
        help="episode_*.hdf5 가 있는 디렉토리",
    )
    args = parser.parse_args()

    qpos = load_qpos_from_episode(args.ep_dir, args.ep_idx)
    print(f"[INFO] Loaded qpos from episode_{args.ep_idx}.hdf5, shape={qpos.shape}")

    plot_joints(qpos, title_prefix=f"episode_{args.ep_idx} ")


if __name__ == "__main__":
    main()
