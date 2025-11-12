#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import h5py
import numpy as np
from typing import Optional, Tuple
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
            # 폴더 이름이 4자리_4자리 형태인지 확인
            name = child.name
            try:
                datetime.strptime(name, "%m%d_%H%M")
                candidates.append(child)
            except ValueError:
                pass

    if not candidates:
        return None

    # 이름 기준으로 정렬하면 시간 순서가 됨
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
                       action: np.ndarray,
                       orig_len: int,
                       T_pad: int,
                       truncated: bool):
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

        # action
        h.create_dataset("/action", data=action, dtype="float64")

        # meta
        meta = h.create_group("/meta")
        meta.create_dataset("orig_len", data=np.array(orig_len, dtype=np.int64))
        meta.create_dataset("T_pad", data=np.array(T_pad, dtype=np.int64))
        meta.create_dataset("pad_starts_at", data=np.array(pad_starts_at, dtype=np.int64))
        meta.create_dataset("truncated", data=np.array(truncated, dtype=np.bool_))

    return out_path

def convert_streaming(input_path: str,
                      output_dir: str,
                      target_len: Optional[int] = None,
                      truncate: bool = False):
    """
    메모리에 전부 올리지 않고 /data/demo_k 를 하나씩 읽어서 바로 episode로 저장.
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

        # 먼저 모든 demo 길이를 한 번 훑어서 max 길이 구함 (메모리 거의 안 듦)
        lengths = []
        for demo_key in demo_keys:
            grp = data_grp[demo_key]
            # 우리가 기록한 이름: joints, image_front, image_top
            if "joints" not in grp:
                print(f"[WARN] {demo_key} has no 'joints', skip")
                lengths.append(0)
                continue
            T = grp["joints"].shape[0]
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

            if "joints" not in grp:
                print(f"[SKIP] {demo_key}: no joints")
                continue

            joints = grp["joints"][()]  # (T, 6)
            # 카메라 이름: image_front / image_top 으로 저장했으니 여기서 매칭
            if "image_front" not in grp or "image_top" not in grp:
                print(f"[WARN] {demo_key}: missing image_front or image_top, skip")
                continue

            img_front = grp["image_front"][()]  # (T, H, W, 3)
            img_top   = grp["image_top"][()]

            # observations/qpos = joints
            qpos = joints.astype(np.float64)
            # action = qpos 그대로
            action = qpos.copy()

            orig_T = qpos.shape[0]

            qpos_p      = pad_repeat_last(qpos,      T_pad)
            img_front_p = pad_repeat_last(img_front, T_pad)
            img_top_p   = pad_repeat_last(img_top,   T_pad)
            action_p    = pad_repeat_last(action,    T_pad)

            ep_name = f"episode_{idx}.hdf5"
            out_path = os.path.join(output_dir, ep_name)
            truncated_flag = (orig_T > T_pad)

            write_episode_hdf5(out_path,
                               qpos_p,
                               img_front_p,
                               img_top_p,
                               action_p,
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

    print("[DONE] conversion complete. T_pad =", T_pad)

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
        output_dir = os.path.join(latest_dir, "episodes") if args.output is None else args.output
    else:
        input_path = args.input
        output_dir = args.output if args.output is not None else os.path.join(os.path.dirname(input_path), "episodes")

    print(f"[INFO] input  = {input_path}")
    print(f"[INFO] output = {output_dir}")

    convert_streaming(input_path, output_dir,
                      target_len=args.target_len,
                      truncate=args.truncate)

if __name__ == "__main__":
    main()
