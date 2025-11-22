#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare demo joints (d0~d5) and ACT inference joints (p0~p5)
from the latest act_infer_*.csv file.

- LOG_DIR 안에서 가장 최신 CSV를 자동으로 찾음
- d0~d5 : infer 시점에서 함께 로깅한 demo joint 값
- p0~p5 : 같은 step에서 policy가 낸 predicted joint 값
- 둘 사이의 MAE / MSE를 joint별로 출력하고, 궤적을 플롯으로 보여줌
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 경로 설정
# -----------------------------
LOG_DIR = "/home/eunseop/nrs_lab2/analysis_logs"


def find_latest_csv(log_dir: str) -> str:
    """log_dir 안에서 act_infer_*.csv 중 가장 최신 파일 하나 반환"""
    csv_list = glob.glob(os.path.join(log_dir, "act_infer_*.csv"))
    if not csv_list:
        raise FileNotFoundError(f"No act_infer_*.csv found in {log_dir}")
    csv_list.sort()  # 파일명 기준 정렬 (시간 문자열이 포함돼 있으니 마지막이 최신)
    return csv_list[-1]


def main():
    # 1) 최신 CSV 불러오기
    csv_path = find_latest_csv(LOG_DIR)
    print(f"[INFO] Using CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2) CSV에서 demo / pred 분리 (d0~d5, p0~p5)
    demo_cols = [f"d{i}" for i in range(6)]
    pred_cols = [f"p{i}" for i in range(6)]

    # 혹시라도 컬럼이 없으면 에러 메시지 출력
    for c in demo_cols + pred_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in CSV: {csv_path}")

    demo = df[demo_cols].to_numpy()  # (N, 6)
    pred = df[pred_cols].to_numpy()  # (N, 6)

    N = demo.shape[0]
    print(f"[INFO] CSV steps N = {N}")

    # 3) joint별 오차 통계 출력 (NaN은 무시)
    diff = pred - demo  # (N, 6)

    # NaN이 있을 수 있으므로 nanmean 사용
    mse = np.nanmean(diff ** 2, axis=0)
    mae = np.nanmean(np.abs(diff), axis=0)

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
