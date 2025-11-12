#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
C++ ActDataRecorder가 만든 hdf5를 ACT 학습용 episode_*.hdf5 로 변환하는 스크립트.

입력 hdf5 구조 (네 C++ 코드 기준):
  ~/nrs_lab2/datasets/ACT/<run_id>/act_data.hdf5
  /data/demo_0/joints       -> (T, 6) float64
  /data/demo_0/ft           -> (T, 3) float32   (fx=0, fy=0, fz만 저장됨)
  /data/demo_0/image_front  -> (T, H, W, 3) uint8  (front camera)
  /data/demo_0/image_top    -> (T, H, W, 3) uint8  (top camera)

출력 hdf5 구조 (ACT 스타일):
  /observations/qpos                    (T_pad, 6) float64
  /observations/qvel                    (T_pad, 6) float64  ← 0으로 채움
  /observations/images/cam_head         (T_pad, H, W, 3) uint8  ← image_top 매핑
  /observations/images/cam_front        (T_pad, H, W, 3) uint8  ← image_front 매핑
  /observations/is_pad                  (T_pad,) bool
  /observations/ft                      (T_pad, 3) float32     ← 있으면

  /action                               (T_pad, 6) float64     ← qpos 그대로

  /meta/orig_len
  /meta/T_pad
  /meta/pad_starts_at
  /meta/truncated

기본 동작:
- ~/nrs_lab2/datasets/ACT/ 아래 폴더들 중에서 수정 시간이 가장 최신인 폴더를 골라서
  그 안의 act_data.hdf5 를 입력으로 사용함.
- 출력은 같은 폴더 안의 episodes/ 밑에 episode_0.hdf5 ... 형태로 생성.
"""

import os
import argparse
import h5py
import numpy as np
from typing import Dict, Any


# ---------------------------------------------------------------------
# 유틸: 가장 최신 run 디렉토리 찾기
# ---------------------------------------------------------------------
def find_latest_run_dir(base_dir: str) -> str:
    """
    base_dir 아래에 있는 하위 디렉토리들 중에서
    수정시간(mtime)이 가장 최근인 디렉토리를 골라서 반환.
    """
    base_dir = os.path.expanduser(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"{base_dir} 디렉토리가 없습니다.")

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"{base_dir} 아래에 하위 폴더가 없습니다.")

    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


# ---------------------------------------------------------------------
# 패딩 함수
# ---------------------------------------------------------------------
def pad_repeat_last(arr: np.ndarray, target_len: int) -> np.ndarray:
    """길이가 짧으면 마지막 프레임을 반복하고, 길면 자른다."""
    T = arr.shape[0]
    if T == target_len:
        return arr
    if T == 0:
        raise ValueError("Cannot pad empty array")
    if T > target_len:
        return arr[:target_len]
    pad_cnt = target_len - T
    last = arr[-1:, ...]
    pad_block = np.repeat(last, pad_cnt, axis=0)
    return np.concatenate([arr, pad_block], axis=0)


# ---------------------------------------------------------------------
# 입력 hdf5 로드: C++ 레코더 포맷에 맞춤
# ---------------------------------------------------------------------
def load_demos(input_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    /data/demo_x 밑에서 joints, image_front, image_top, (ft) 를 읽어서
    파이썬 dict로 돌려준다.
    """
    demos: Dict[str, Dict[str, np.ndarray]] = {}

    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("input hdf5에 '/data' 그룹이 없습니다.")

        data_grp = f["data"]
        for demo_key in sorted(data_grp.keys()):
            grp = data_grp[demo_key]

            # 필수 키 확인
            required = ["joints", "image_front", "image_top"]
            missing = [k for k in required if k not in grp]
            if missing:
                print(f"[WARN] {demo_key} 에 {missing} 없음 -> 스킵")
                continue

            joints = np.array(grp["joints"])            # (T, 6)
            img_front = np.array(grp["image_front"])    # (T, H, W, 3)
            img_top = np.array(grp["image_top"])        # (T, H, W, 3)

            if "ft" in grp:
                ft = np.array(grp["ft"])                # (T, 3)
            else:
                ft = None

            demos[demo_key] = {
                "joints": joints,
                "image_front": img_front,
                "image_top": img_top,
                "ft": ft,
            }

    if not demos:
        raise ValueError("유효한 demo를 하나도 못 읽었습니다.")
    return demos


# ---------------------------------------------------------------------
# episode_*.hdf5 하나 저장
# ---------------------------------------------------------------------
def write_episode_hdf5(
    out_path: str,
    qpos: np.ndarray,
    cam_head: np.ndarray,
    cam_front: np.ndarray,
    action: np.ndarray,
    orig_len: int,
    T_pad: int,
    truncated: bool,
    ft: np.ndarray | None = None,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 패딩 여부 마스크
    is_pad = np.zeros((T_pad,), dtype=bool)
    pad_starts_at = -1
    if orig_len < T_pad:
        is_pad[orig_len:] = True
        pad_starts_at = int(orig_len)

    with h5py.File(out_path, "w") as h:
        h.attrs["sim"] = False

        # 관측 값
        h.create_dataset("/observations/qpos", data=qpos, dtype="float64")

        qvel = np.zeros_like(qpos, dtype=np.float64)
        h.create_dataset("/observations/qvel", data=qvel, dtype="float64")

        h.create_dataset(
            "/observations/images/cam_head",
            data=cam_head,
            dtype="uint8",
            chunks=True,
            compression="lzf",
        )
        h.create_dataset(
            "/observations/images/cam_front",
            data=cam_front,
            dtype="uint8",
            chunks=True,
            compression="lzf",
        )

        h.create_dataset("/observations/is_pad", data=is_pad, dtype="bool")

        if ft is not None:
            h.create_dataset("/observations/ft", data=ft, dtype="float32")

        # action = qpos 복사
        h.create_dataset("/action", data=action, dtype="float64")

        # 메타데이터
        meta = h.create_group("/meta")
        meta.create_dataset("orig_len", data=np.array(orig_len, dtype=np.int64))
        meta.create_dataset("T_pad", data=np.array(T_pad, dtype=np.int64))
        meta.create_dataset("pad_starts_at", data=np.array(pad_starts_at, dtype=np.int64))
        meta.create_dataset("truncated", data=np.array(truncated, dtype=np.bool_))

    return out_path, pad_starts_at


# ---------------------------------------------------------------------
# 전체 변환
# ---------------------------------------------------------------------
def convert(
    input_path: str,
    output_dir: str,
    pad_mode: str = "repeat_last",
    target_len: int | None = None,
    truncate: bool = False,
):
    demos = load_demos(input_path)

    # 데모별 길이
    lengths = {k: demos[k]["joints"].shape[0] for k in demos}
    T_max = max(lengths.values())

    # 최종 길이 결정
    if target_len is None:
        T_pad = T_max
    else:
        T_pad = target_len if truncate else max(T_max, target_len)

    manifest: Dict[str, Any] = {
        "input": input_path,
        "output_dir": output_dir,
        "pad_mode": pad_mode,
        "T_pad": T_pad,
        "truncate": truncate,
        "episodes": [],
    }

    for idx, demo_key in enumerate(sorted(demos.keys())):
        joints = demos[demo_key]["joints"]          # (T, 6)
        img_front = demos[demo_key]["image_front"]  # (T, H, W, 3)
        img_top = demos[demo_key]["image_top"]      # (T, H, W, 3)
        ft = demos[demo_key]["ft"]                  # (T, 3) or None

        qpos = joints.astype(np.float64)
        action = qpos.copy()

        if pad_mode != "repeat_last":
            raise NotImplementedError("현재는 repeat_last 패딩만 지원합니다.")

        orig_T = int(qpos.shape[0])

        qpos_p = pad_repeat_last(qpos, T_pad)
        cam_head_p = pad_repeat_last(img_top, T_pad)     # top -> head
        cam_front_p = pad_repeat_last(img_front, T_pad)  # front -> front
        action_p = pad_repeat_last(action, T_pad)

        if ft is not None:
            ft_p = pad_repeat_last(ft, T_pad).astype(np.float32)
        else:
            ft_p = None

        ep_name = f"episode_{idx}.hdf5"
        out_path = os.path.join(output_dir, ep_name)
        truncated_flag = orig_T > T_pad

        out_path, pad_starts_at = write_episode_hdf5(
            out_path,
            qpos_p,
            cam_head_p,
            cam_front_p,
            action_p,
            orig_len=orig_T,
            T_pad=T_pad,
            truncated=truncated_flag,
            ft=ft_p,
        )

        manifest["episodes"].append(
            {
                "demo_key": demo_key,
                "episode_file": out_path,
                "orig_T": orig_T,
                "T_padded": int(qpos_p.shape[0]),
                "pad_from": (orig_T if orig_T < T_pad else None),
                "truncated_from": (orig_T if orig_T > T_pad else None),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    import json
    with open(os.path.join(output_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    return manifest


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    # 1) ACT 루트 디렉토리
    ACT_BASE = "~/nrs_lab2/datasets/ACT"

    # 2) 그 안에서 가장 최신 run 디렉토리 찾기
    try:
        latest_dir = find_latest_run_dir(ACT_BASE)
    except FileNotFoundError:
        # 없으면 그냥 ACT_BASE 를 입력으로 씀
        latest_dir = os.path.expanduser(ACT_BASE)

    # 3) 기본 입출력 경로
    default_input = os.path.join(latest_dir, "act_data.hdf5")
    default_output = os.path.join(latest_dir, "episodes")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=default_input,
                        help=f"input hdf5 path (default: {default_input})")
    parser.add_argument("-o", "--output", default=default_output,
                        help=f"output dir to save episode_*.hdf5 (default: {default_output})")
    parser.add_argument("--pad", choices=["repeat_last"], default="repeat_last")
    parser.add_argument("--target-len", type=int, default=None,
                        help="target episode length. if not set, use max length among demos")
    parser.add_argument("--truncate", action="store_true",
                        help="if set, long demos will be truncated to target-len")

    args = parser.parse_args()

    print(f"[INFO] input  = {args.input}")
    print(f"[INFO] output = {args.output}")

    manifest = convert(
        args.input,
        args.output,
        args.pad,
        args.target_len,
        args.truncate,
    )

    print("Conversion complete. T_pad =", manifest["T_pad"])
    for ep in manifest["episodes"]:
        if ep["pad_from"] is None:
            pad_msg = "패딩 없음"
        else:
            pad_msg = f"패딩={ep['pad_from']}~{ep['T_padded']-1}"
        print(
            f"- {ep['demo_key']} -> {ep['episode_file']} "
            f"(실제={ep['orig_T']}, 최종={ep['T_padded']}, {pad_msg}, 잘림={ep['truncated_from']})"
        )


if __name__ == "__main__":
    main()

