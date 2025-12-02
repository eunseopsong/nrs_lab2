import os
from datetime import datetime

import torch

# ---------------------------------------------------------------------
# 1) 가장 최신 타임스탬프 폴더에서 policy_best.ckpt 선택
# ---------------------------------------------------------------------
CKPT_ROOT = "/home/eunseop/nrs_lab2/checkpoints/ur10e_swing"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_latest_timestamped_subdir(root_dir: str):
    """
    root_dir 아래에서 이름이 '%m%d_%H%M' 형식인 폴더들 중
    가장 최신 서브폴더 경로를 반환.
    """
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for name in os.listdir(root_dir):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        try:
            datetime.strptime(name, "%m%d_%H%M")  # 형식 체크
            candidates.append((name, sub))
        except ValueError:
            continue

    if not candidates:
        return None

    # 문자열 기준 내림차순 정렬 → 가장 최근이 맨 앞
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def get_ckpt_path(ckpt_root: str):
    # 1) root 바로 아래 policy_best.ckpt 있으면 그걸 사용
    root_level_ckpt = os.path.join(ckpt_root, "policy_best.ckpt")
    if os.path.exists(root_level_ckpt):
        return root_level_ckpt

    # 2) 아니면 timestamp 서브폴더들 중 최신 것을 사용
    latest_subdir = find_latest_timestamped_subdir(ckpt_root)
    if latest_subdir is None:
        raise FileNotFoundError(f"No timestamped subdirectories found under {ckpt_root}")

    ckpt_path = os.path.join(latest_subdir, "policy_best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"policy_best.ckpt not found in latest subdir: {ckpt_path}")
    return ckpt_path


def main():
    ckpt_path = get_ckpt_path(CKPT_ROOT)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # state_dict 형태 정리
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    print("\n=== PARAMETER SHAPES IN STATE_DICT ===")
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name:60s} {tuple(tensor.shape)}")

    # 간단히 "마지막 레이어" 후보 몇 개 출력 (2D weight 중 마지막 것들)
    print("\n=== CANDIDATE FINAL LINEAR LAYERS (2D weight, 마지막 몇 개) ===")
    linear_weights = [(n, t) for n, t in state_dict.items()
                      if isinstance(t, torch.Tensor) and t.ndim == 2]

    for name, tensor in linear_weights[-10:]:  # 뒤에서부터 10개
        print(f"{name:60s} {tuple(tensor.shape)}")

    # 가장 큰 out_features(=첫 번째 차원)를 갖는 weight도 참고용으로 출력
    if linear_weights:
        max_w = max(linear_weights, key=lambda p: p[1].shape[0])
        print("\n[INFO] Max out_features among Linear-like weights:")
        print(f"  {max_w[0]}  shape={tuple(max_w[1].shape)}")


if __name__ == "__main__":
    main()
