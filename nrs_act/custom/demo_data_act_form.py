#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import h5py
import numpy as np
from typing import Optional
from datetime import datetime
from pathlib import Path

ROOT_DEFAULT = "/home/eunseop/nrs_lab2/datasets/ACT"

def find_latest_folder(root_dir: str) -> Optional[str]:
    """
    /home/.../ACT 아래에 MMDD_HHMM 형태 폴더들 중 가장 최신을 고른다.
    """
    p = Path(root_dir)
    if not p.exists():
        return None

    candidates = []
    for child in p.iterdir():
        if child.is_dir():
            name = child.name
            try:
                datetime.strptime(name, "%m%d_%H%M")
                candidates.append(child)
            except ValueError:
                pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.name, reverse=True)
    return str(candidates[0])

def pad_repeat_last(arr: np.ndarray, target_len: int) -> np.ndarray:
    T = arr.shape[0]
    if T == target_len:
        return arr
    if T == 0:
        raise ValueError("Cannot pad an empty array with repeat_last mode.")
    if T > target_len:
        # 잘라내기
        return arr[:target_len]
    pad_count = target_len - T
    last = arr[-1:,...]
    pad_block = np.repeat(last, pad_count, axis=0)
    return np.concatenate([arr, pad_block], axis=0)

def write_episode_hdf5(out_path: str,
                       qpos: np.ndarray,
                       cam_front: np.ndarray,
                       cam_head: np.ndarray,
                       action_joints: np.ndarray,
                       action_ft: np.ndarray,
                       orig_len: int,
                       T_pad: int,
                       truncated: bool):
    """
    하나의 demo를 ACT-style episode_x.hdf5로 저장.

    - /observations/qpos          : (T_pad, 6)   <- obs_joints
    - /observations/qvel          : (T_pad, 6)   <- zeros
    - /observations/images/cam_*  : (T_pad, H, W, 3)
    - /observations/is_pad        : (T_pad, )

    - /action/joints              : (T_pad, 6)   <- action_joints
    - /action/ft                  : (T_pad, 3)   <- action_ft

    ※ ft 는 observation 에 넣지 않음
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # padding mask
    is_pad = np.zeros((T_pad,), dtype=bool)
    pad_starts_at = -1
    if orig_len < T_pad:
        is_pad[orig_len:] = True
        pad_starts_at = int(orig_len)

    with h5py.File(out_path, "w") as h:
        h.attrs["sim"] = False

        # observations
        h.create_dataset("/observations/qpos", data=qpos, dtype="float64")
        qvel = np.zeros_like(qpos, dtype=np.float64)
        h.create_dataset("/observations/qvel", data=qvel, dtype="float64")

        # 이미지
        h.create_dataset("/observations/images/cam_front",
                         data=cam_front,
                         dtype="uint8",
                         chunks=True,
                         compression="lzf")
        h.create_dataset("/observations/images/cam_head",
                         data=cam_head,
                         dtype="uint8",
                         chunks=True,
                         compression="lzf")

        h.create_dataset("/observations/is_pad", data=is_pad, dtype="bool")

        # action: joints / ft 를 분리해서 저장
        act_grp = h.create_group("/action")
        act_grp.create_dataset("joints", data=action_joints.astype(np.float64), dtype="float64")
        act_grp.create_dataset("ft",     data=action_ft.astype(np.float64),     dtype="float64")

        # meta
        meta = h.create_group("/meta")
        meta.create_dataset("orig_len",      data=np.array(orig_len, dtype=np.int64))
        meta.create_dataset("T_pad",         data=np.array(T_pad, dtype=np.int64))
        meta.create_dataset("pad_starts_at", data=np.array(pad_starts_at, dtype=np.int64))
        meta.create_dataset("truncated",     data=np.array(truncated, dtype=np.bool_))

    return out_path

def convert_streaming(input_path: str,
                      output_dir: str,
                      target_len: Optional[int] = None,
                      truncate: bool = False):
    """
    메모리에 전부 올리지 않고 /data/demo_k 를 하나씩 읽어서 바로 episode로 저장.
    ur10e_act_data_gen.cpp 가 만든 새 HDF5 구조를 그대로 사용:

      - obs_joints      : (T, 6)
      - action_joints   : (T, 6)
      - action_ft       : (T, 3)
      - obs_image_front : (T, H, W, 3)
      - obs_image_top   : (T, H, W, 3)
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "input": input_path,
        "output_dir": output_dir,
        "pad_mode": "repeat_last",
        "truncate": truncate,
        "episodes": []
    }

    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("No 'data' group in input file.")
        data_grp = f["data"]
        demo_keys = sorted(list(data_grp.keys()))
        if len(demo_keys) == 0:
            raise ValueError("No demos found in input.")

        # 먼저 모든 demo 길이를 한 번 훑어서 max 길이 구함 (obs_joints 기준)
        lengths = []
        for demo_key in demo_keys:
            grp = data_grp[demo_key]
            if "obs_joints" not in grp:
                print(f"[WARN] {demo_key} has no 'obs_joints', skip")
                lengths.append(0)
                continue
            T = grp["obs_joints"].shape[0]
            lengths.append(T)

        T_max = max(lengths)

        # 최종 pad 길이 결정
        if target_len is None:
            T_pad = T_max
        else:
            T_pad = target_len if truncate else max(T_max, target_len)

        manifest["T_pad"] = int(T_pad)

        # 이제 실제 변환
        for idx, demo_key in enumerate(demo_keys):
            grp = data_grp[demo_key]

            # 필수 키 체크
            required_keys = [
                "obs_joints",
                "action_joints",
                "action_ft",
                "obs_image_front",
                "obs_image_top",
            ]
            missing = [k for k in required_keys if k not in grp]
            if missing:
                print(f"[SKIP] {demo_key}: missing {missing}")
                continue

            obs_joints    = grp["obs_joints"][()]       # (T, 6)
            action_joints = grp["action_joints"][()]    # (T, 6)
            action_ft     = grp["action_ft"][()]        # (T, 3)
            img_front     = grp["obs_image_front"][()]  # (T, H, W, 3)
            img_top       = grp["obs_image_top"][()]    # (T, H, W, 3)

            # 길이 불일치 방어: 최소 길이에 맞춰 자르기
            T_list = [
                obs_joints.shape[0],
                action_joints.shape[0],
                action_ft.shape[0],
                img_front.shape[0],
                img_top.shape[0],
            ]
            T_min = min(T_list)
            if T_min < 1:
                print(f"[SKIP] {demo_key}: too short (T_min={T_min})")
                continue

            if obs_joints.shape[0] != T_min:
                obs_joints = obs_joints[:T_min]
            if action_joints.shape[0] != T_min:
                action_joints = action_joints[:T_min]
            if action_ft.shape[0] != T_min:
                action_ft = action_ft[:T_min]
            if img_front.shape[0] != T_min:
                img_front = img_front[:T_min]
            if img_top.shape[0] != T_min:
                img_top = img_top[:T_min]

            # observations/qpos = obs_joints
            qpos = obs_joints.astype(np.float64)

            orig_T = qpos.shape[0]

            # pad
            qpos_p          = pad_repeat_last(qpos,          T_pad)
            img_front_p     = pad_repeat_last(img_front,     T_pad)
            img_top_p       = pad_repeat_last(img_top,       T_pad)
            action_joints_p = pad_repeat_last(action_joints, T_pad)
            action_ft_p     = pad_repeat_last(action_ft,     T_pad)

            ep_name = f"episode_{idx}.hdf5"
            out_path = os.path.join(output_dir, ep_name)
            truncated_flag = (orig_T > T_pad)

            write_episode_hdf5(out_path,
                               qpos_p,
                               img_front_p,
                               img_top_p,
                               action_joints_p,
                               action_ft_p,
                               orig_len=orig_T,
                               T_pad=T_pad,
                               truncated=truncated_flag)

            manifest["episodes"].append({
                "demo_key": demo_key,
                "episode_file": out_path,
                "orig_T": int(orig_T),
                "T_padded": int(T_pad),
                "pad_from": (int(orig_T) if orig_T < T_pad else None),
                "truncated_from": (int(orig_T) if orig_T > T_pad else None)
            })

            print(f"[OK] {demo_key} -> {out_path} (orig={orig_T}, final={T_pad})")

    # manifest 저장
    import json
    with open(os.path.join(output_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("[DONE] conversion complete. T_pad =", manifest["T_pad"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT_DEFAULT,
                        help="ACT 데이터 루트(/home/.../datasets/ACT)")
    parser.add_argument("--input", "-i", default=None,
                        help="직접 hdf5 지정하고 싶을 때")
    parser.add_argument("--output", "-o", default=None,
                        help="출력 디렉터리 직접 지정")
    parser.add_argument("--target-len", type=int, default=None)
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    # 1) input 없으면 최신 폴더 찾아서 거기 hdf5 사용
    if args.input is None:
        latest_dir = find_latest_folder(args.root)
        if latest_dir is None:
            raise FileNotFoundError(f"No dated folder found under {args.root}")
        input_path = os.path.join(latest_dir, "act_data.hdf5")
        # 기본 출력 폴더를 episodes_ft 로 설정
        default_subdir = "episodes_ft"
        output_dir = args.output if args.output is not None else os.path.join(latest_dir, default_subdir)
    else:
        input_path = args.input
        if args.output is not None:
            output_dir = args.output
        else:
            # input hdf5와 같은 상위 폴더 아래에 episodes_ft 생성
            base_dir = os.path.dirname(input_path)
            output_dir = os.path.join(base_dir, "episodes_ft")

    print(f"[INFO] input  = {input_path}")
    print(f"[INFO] output = {output_dir}")

    convert_streaming(input_path, output_dir,
                      target_len=args.target_len,
                      truncate=args.truncate)

if __name__ == "__main__":
    main()
