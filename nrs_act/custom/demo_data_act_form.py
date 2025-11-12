#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
사용법:
1) 가장 긴 데모 길이에 맞춰 패딩(잘라내기 X)
python3 demo_data_act_form.py \
  --input  <input hdf5 path> \
  --output /home/vision/dualarm_ws/src/dualarm_il/data \
  --pad repeat_last

2) 원하는 스텝으로 맞추기(길면 자르고, 짧으면 패딩)
python3 demo_data_act_form.py \
  --input <input hdf5 path> \
  --output /home/vision/dualarm_ws/src/dualarm_il/data \
  --target-len <step len> --truncate \
  --pad repeat_last

3) 최소 숫자 스텝으로(짧은 데모만 패딩, 긴 데모는 그대로)
python3 demo_data_act_form.py \
  --input  <input hdf5 path> \
  --output /home/vision/dualarm_ws/src/dualarm_il/data \
  --target-len <step len> \
  --pad repeat_last

4) 원하는 step까지 padding
python3 demo_data_act_form.py \
  --input /home/vision/dualarm_ws/src/dualarm_il/data/0818_2144/common_data.hdf5 \
  --output /home/vision/dualarm_ws/src/dualarm_il/data/ACT \
  --target-len <step_num> \
  --pad repeat_last

"""

"""
demo_data_act_form.py

Convert `common_data.hdf5` (from dualarm_data_gen.py) into ACT-style per-episode HDF5 files.

Input format (per demo_k):
  data/demo_k/observations/
    - joint_L:     (T, 6) float64
    - joint_R:     (T, 6) float64
    - cam_head:     (T, 480, 640, 3) uint8
    - cam_front:    (T, 480, 640, 3) uint8
    - (TCP_pose_*, TCP_quat_* are ignored)

Output format (per episode_i.hdf5):
  /observations/qpos            -> (T_pad, 12) float64
  /observations/images/cam_head  -> (T_pad, 480, 640, 3) uint8
  /observations/images/cam_front  -> (T_pad, 480, 640, 3) uint8
    /action                       -> (T_pad, 12) float64
Also writes:
  /observations/is_pad          -> (T_pad,) bool
  /meta/orig_len (int), /meta/T_pad (int), /meta/pad_starts_at (int, -1 if none), /meta/truncated (bool)
"""
import os
import argparse
import h5py
import numpy as np
from typing import Dict

def pad_repeat_last(arr: np.ndarray, target_len: int) -> np.ndarray:
    T = arr.shape[0]
    if T == target_len:
        return arr
    if T == 0:
        raise ValueError("Cannot pad an empty array with repeat_last mode.")
    if T > target_len:
        # 잘라내기(앞에서 target_len까지만)
        return arr[:target_len]
    # 패딩하기(마지막 프레임 반복)
    pad_count = target_len - T
    last = arr[-1:,...]
    pad_block = np.repeat(last, pad_count, axis=0)
    return np.concatenate([arr, pad_block], axis=0)

def load_common(input_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    demos = {}
    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("No 'data' group in input file.")
        data_grp = f["data"]
        for demo_key in sorted(data_grp.keys()):
            grp = data_grp[demo_key]
            
            # observations 그룹이 있는지 확인하고, 없으면 직접 demo 그룹에서 데이터 읽기
            if "observations" in grp:
                obs = grp["observations"]
            else:
                obs = grp  # 직접 demo 그룹에서 데이터 읽기
            
            # 필요한 데이터셋들이 있는지 확인
            required_keys = ["joint_L", "joint_R", "image_H", "image_F"]
            missing_keys = [k for k in required_keys if k not in obs]
            if missing_keys:
                print(f"Warning: Missing datasets {missing_keys} in {demo_key}, skipping...")
                continue
                
            demos[demo_key] = {
                "joint_L": np.array(obs["joint_L"]),
                "joint_R": np.array(obs["joint_R"]),
                "image_H": np.array(obs["image_H"]),
                "image_F": np.array(obs["image_F"]),
            }
    
    if not demos:
        raise ValueError("No demos found in input.")
    return demos

def write_episode_hdf5(out_path: str, qpos: np.ndarray, cam_head: np.ndarray,cam_front: np.ndarray, action: np.ndarray,
                       orig_len: int, T_pad: int, truncated: bool):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 패딩 정보 마스크 생성
    is_pad = np.zeros((T_pad,), dtype=bool)
    pad_starts_at = -1
    if orig_len < T_pad:
        is_pad[orig_len:] = True
        pad_starts_at = int(orig_len)

    with h5py.File(out_path, "w") as h:
        # 루트 속성 추가
        h.attrs["sim"] = False

        # Observations
        h.create_dataset("/observations/qpos", data=qpos, dtype="float64")

        qvel = np.zeros(qpos.shape, dtype=np.float64)   # (T_pad, 12) zeros
        h.create_dataset("/observations/qvel", data=qvel, dtype="float64")

        h.create_dataset("/observations/images/cam_head", data=cam_head, dtype="uint8", chunks=True, compression="lzf")
        h.create_dataset("/observations/images/cam_front", data=cam_front, dtype="uint8", chunks=True, compression="lzf")
        h.create_dataset("/observations/is_pad", data=is_pad, dtype="bool")

        # Action
        h.create_dataset("/action", data=action, dtype="float64")

        # Meta
        meta = h.create_group("/meta")
        meta.create_dataset("orig_len", data=np.array(orig_len, dtype=np.int64))
        meta.create_dataset("T_pad", data=np.array(T_pad, dtype=np.int64))
        meta.create_dataset("pad_starts_at", data=np.array(pad_starts_at, dtype=np.int64))  # -1 if no padding
        meta.create_dataset("truncated", data=np.array(truncated, dtype=np.bool_))

    return out_path, pad_starts_at

def convert(input_path: str, output_dir: str, pad_mode: str = "repeat_last",
            target_len: int | None = None, truncate: bool = False):
    demos = load_common(input_path)
    lengths = {k: demos[k]["joint_L"].shape[0] for k in demos}
    T_max = max(lengths.values())

    # 최종 목표 길이 결정
    if target_len is None:
        T_pad = T_max
    else:
        T_pad = target_len if truncate else max(T_max, target_len)

    manifest = {"input": input_path, "output_dir": output_dir,
                "pad_mode": pad_mode, "T_pad": T_pad, "truncate": truncate, "episodes": []}

    for idx, demo_key in enumerate(sorted(demos.keys())):
        joint_L = demos[demo_key]["joint_L"]
        joint_R = demos[demo_key]["joint_R"]
        cam_head = demos[demo_key]["image_H"]
        cam_front = demos[demo_key]["image_F"]

        qpos = np.concatenate([joint_L, joint_R], axis=1).astype(np.float64)
        action = qpos.copy()

        if pad_mode != "repeat_last":
            raise NotImplementedError(f"pad_mode '{pad_mode}' not implemented.")

        orig_T = int(qpos.shape[0])
        # 길이 맞추기 (길면 자르고, 짧으면 마지막 프레임 반복)
        qpos_p   = pad_repeat_last(qpos, T_pad)
        cam_head_p    = pad_repeat_last(cam_head, T_pad)
        cam_front_p = pad_repeat_last(cam_front, T_pad)
        action_p = pad_repeat_last(action, T_pad)

        ep_name = f"episode_{idx}.hdf5"
        out_path = os.path.join(output_dir, ep_name)
        truncated_flag = (orig_T > T_pad)
        out_path, pad_starts_at = write_episode_hdf5(
            out_path, qpos_p, cam_head_p, cam_front_p, action_p,
            orig_len=orig_T, T_pad=T_pad, truncated=truncated_flag
        )

        manifest["episodes"].append({
            "demo_key": demo_key,
            "episode_file": out_path,
            "orig_T": orig_T,
            "T_padded": int(qpos_p.shape[0]),
            "pad_from": (orig_T if orig_T < T_pad else None),          # padding 시작 인덱스
            "truncated_from": (orig_T if orig_T > T_pad else None)      # 잘리기 전 길이
        })

    os.makedirs(output_dir, exist_ok=True)
    import json
    with open(os.path.join(output_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    return manifest

def main():
    parser = argparse.ArgumentParser()
    # 기본 경로 상수
    # DEFAULT_INPUT = "/home/vision/dualarm_ws/src/dualarm_il/data/0818_2122/common_data.hdf5"
    DEFAULT_INPUT = "/home/vision/isaacsim/chkwon_isaac/imitation_learning/data/1015_1257/common_data.hdf5"
    
    DEFAULT_OUTPUT = "/home/vision/isaacsim/chkwon_isaac/imitation_learning/data/1015_1257"

    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to common_data.hdf5 (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Directory to write episode_*.hdf5 (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument("--pad", choices=["repeat_last"], default="repeat_last")
    parser.add_argument(
        "--target-len", type=int, default=None,
        help="Final episode length. If not set, uses max length across demos."
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="If set, and target length is shorter than some episodes, truncate them."
    )
    args = parser.parse_args()

    # (선택) 사용자 편의를 위한 경로 표시
    print(f"[INFO] input = {args.input}")
    print(f"[INFO] output = {args.output}")

    manifest = convert(args.input, args.output, args.pad, args.target_len, args.truncate)
    print("Conversion complete. T_pad =", manifest["T_pad"])
    # 콘솔에서 패딩 구간/잘림 즉시 확인
    for ep in manifest["episodes"]:
        pad_from = ep["pad_from"]
        if pad_from is None:
            pad_msg = "패딩 없음"
        else:
            pad_msg = f"패딩={pad_from}~{ep['T_padded']-1}"
        print(f"- {ep['demo_key']} -> {ep['episode_file']} "
              f"(실제={ep['orig_T']}, 최종={ep['T_padded']}, {pad_msg}, 잘림={ep['truncated_from']})")

if __name__ == "__main__":
    main()

