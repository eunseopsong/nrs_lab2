#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare demo joints (from dataset) and ACT inference joints (from CSV).
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) 가장 최신 CSV 찾기
# -----------------------------
LOG_DIR = "/home/eunseop/nrs_lab2/analysis_logs"
EPISODE_DIR = "/home/eunseop/nrs_lab2/datasets/ACT/1114_1643/episodes"


def find_latest_csv(log_dir: str) -> str:
    csv_list = glob.glob(os.path.join(log_dir, "act_infer_*.csv"))
    if not csv_list:
        raise FileNotFoundError(f"No act_infer_*.csv found in {log_dir}")
    csv_list.sort()  # 파일명 기준 정렬 (시간 문자열이 포함돼 있으니 마지막이 최신)
    return csv_list[-1]


def find_longest_episode(episode_dir: str) -> str:
    """(선택) episodes 중 길이 가장 긴 파일 다시 찾기 (node에서 했던 거랑 동일 로직)"""
    import h5py

    best_T = -1
    best_path = None

    for fname in sorted(os.listdir(episode_dir)):
        if not (fname.startswith("episode_") and fname.endswith(".hdf5")):
            continue
        full_path = os.path.join(episode_dir, fname)
        try:
            with h5py.File(full_path, "r") as h:
                q = h["/observations/qpos"]
                T = q.shape[0]
        except Exception as e:
            print(f"[WARN] failed to read {full_path}: {e}")
            continue

        if T > best_T:
            best_T = T
            best_path = full_path

    if best_path is None:
        raise RuntimeError(f"No valid episode_*.hdf5 in {episode_dir}")
    print(f"[INFO] Longest episode: {best_path} (T={best_T})")
    return best_path, best_T


def main():
    # 1) 최신 CSV 불러오기
    csv_path = find_latest_csv(LOG_DIR)
    print(f"[INFO] Using CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2) CSV에서 demo / pred 분리
    demo_cols = [f"d{i}" for i in range(6)]
    pred_cols = [f"p{i}" for i in range(6)]

    demo = df[demo_cols].to_numpy()   # (N, 6)
    pred = df[pred_cols].to_numpy()   # (N, 6)

    N = demo.shape[0]
    print(f"[INFO] CSV steps N = {N}")

    # (선택) episodes의 길이도 확인
    try:
        ep_path, T_ep = find_longest_episode(EPISODE_DIR)
        print(f"[INFO] CSV length vs episode length: {N} vs {T_ep}")
    except Exception as e:
        print(f"[WARN] Episode read skipped: {e}")

    # 3) joint별 오차 통계 출력
    diff = pred - demo
    mse = np.mean(diff ** 2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)

    print("\n=== Per-joint error (rad) ===")
    for j in range(6):
        print(f"joint {j}: MAE = {mae[j]:.4f}, MSE = {mse[j]:.4f}")

    # 4) joint별 궤적 비교 플롯
    t = np.arange(N)

    joint_names = ["j0", "j1", "j2", "j3", "j4", "j5"]

    plt.figure(figsize=(12, 10))
    for j in range(6):
        plt.subplot(3, 2, j + 1)
        plt.plot(t, demo[:, j], label=f"{joint_names[j]} demo", linewidth=1.5)
        plt.plot(t, pred[:, j], label=f"{joint_names[j]} pred", linewidth=1.0, linestyle="--")
        plt.xlabel("step")
        plt.ylabel("rad")
        plt.title(f"{joint_names[j]} (demo vs pred)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
