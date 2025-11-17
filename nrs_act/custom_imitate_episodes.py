#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT training & evaluation script (for UR10e dataset)
- UR10e + 2-camera ACT 세팅
"""

import os
import pickle
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from custom.custom_constants import DT, TASK_CONFIGS
from utils import load_data, compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy


# -------------------------------------------------------------------------
# 유틸: 루트 디렉터리 아래에서 MMDD_HHMM 형식의 가장 최신 폴더 찾기
# -------------------------------------------------------------------------
def find_latest_timestamped_subdir(root_dir: str) -> str | None:
    """
    ckpt_root_dir 아래에서 이름이 '%m%d_%H%M' 형식인 폴더들 중 가장 최신을 반환.
    예) 1116_1203, 1114_1643 등
    """
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for name in os.listdir(root_dir):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        try:
            # 형식 맞는지만 확인 (값은 안 씀)
            datetime.strptime(name, "%m%d_%H%M")
            candidates.append((name, sub))
        except ValueError:
            continue

    if not candidates:
        return None

    # 폴더명 기준 역순 정렬 → 가장 최근 것 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def make_policy(policy_class, policy_config):
    return ACTPolicy(policy_config) if policy_class == "ACT" else CNNMLPPolicy(policy_config)


def make_optimizer(policy_class, policy):
    return policy.configure_optimizers()


def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = (
        image_data.to(device),
        qpos_data.to(device),
        action_data.to(device),
        is_pad.to(device),
    )
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs    = config["num_epochs"]
    ckpt_dir      = config["ckpt_dir"]
    seed          = config["seed"]
    policy_class  = config["policy_class"]
    policy_config = config["policy_config"]
    device        = config["device"]

    set_seed(seed)
    policy    = make_policy(policy_class, policy_config).to(device)
    optimizer = make_optimizer(policy_class, policy)

    train_history, validation_history = [], []
    min_val_loss   = np.inf
    best_ckpt_info = None

    # ------------ 디버그: 모델 구조/파라미터 수 출력 ------------
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[DEBUG] Policy class = {policy_class}, trainable params = {n_params/1e6:.2f}M")

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        # ---------------- Validation ----------------
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = [forward_pass(d, policy, device) for d in val_dataloader]
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary["loss"]

            # 디버그: 첫 에폭에서 loss 구성 요소 한 번 찍기
            if epoch == 0:
                print("[DEBUG] Val dict example:", {k: float(v) for k, v in epoch_summary.items()})

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss

                # ✅ ACT일 때는 DETRVAE(backbone+transformer) state_dict만 따로 저장
                if policy_class == "ACT":
                    best_state = deepcopy(policy.model.state_dict())
                else:
                    best_state = deepcopy(policy.state_dict())

                best_ckpt_info = (epoch, min_val_loss, best_state)

        print(f"Val loss:   {epoch_val_loss:.5f}")

        # ---------------- Training ----------------
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

            # 디버그: 첫 에폭 앞쪽 배치 몇 개 loss 출력
            if epoch == 0 and batch_idx < 3:
                print(f"[DEBUG] Epoch 0, batch {batch_idx}, train loss = {float(loss):.5f}")

        # ---------------- Epoch summary ----------------
        num_batches = batch_idx + 1
        start = num_batches * epoch
        end   = num_batches * (epoch + 1)
        epoch_summary = compute_dict_mean(train_history[start:end])
        print(f"Train loss: {epoch_summary['loss']:.5f}")

        # 100 에폭마다 중간 체크포인트 저장
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            if policy_class == "ACT":
                state_to_save = policy.model.state_dict()
            else:
                state_to_save = policy.state_dict()
            torch.save(state_to_save, ckpt_path)
            print(f"[INFO] Saved intermediate ckpt -> {ckpt_path}")
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    print(f"[INFO] Best epoch = {best_epoch}, min val loss = {min_val_loss:.6f}")
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plt.figure()
        train_values = [x[key].item() for x in train_history]
        val_values   = [x[key].item() for x in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_values)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(val_values)),   val_values,   label="val")
        plt.legend()
        plt.tight_layout()
        plt.title(key)
        plt.savefig(os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png"))
        plt.close()
    print(f"[INFO] Saved plots to {ckpt_dir}")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    # CLI args
    is_eval       = args["eval"]
    task_name     = args["task_name"]
    ckpt_root_dir = args["ckpt_dir"]   # 사용자가 넘기는 것은 "루트" 디렉터리
    policy_class  = args["policy_class"]
    num_epochs    = args["num_epochs"]
    batch_size    = args["batch_size"]
    seed          = args["seed"]
    lr            = args["lr"]
    dataset_dir_override = args.get("dataset_dir", None)

    # ------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------
    if task_name not in TASK_CONFIGS:
        raise KeyError(f"[ERROR] task_name '{task_name}' not found in TASK_CONFIGS.")
    task_config  = TASK_CONFIGS[task_name]
    dataset_dir  = dataset_dir_override or task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len  = task_config.get("episode_len", 100)
    camera_names = task_config["camera_names"]

    print(f"[INFO] task_name      = {task_name}")
    print(f"[INFO] dataset_dir    = {dataset_dir}")
    print(f"[INFO] num_episodes   = {num_episodes}")
    print(f"[INFO] camera_names   = {camera_names}")

    # ------------------------------------------------------------
    # Policy config
    # ------------------------------------------------------------
    state_dim    = 6  # UR10e: 6 DOF
    lr_backbone  = 1e-5
    backbone     = "resnet18"

    if policy_class == "ACT":
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": lr,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    # ---------------- 공통 config ----------------
    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": None,            # 아래에서 채움
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": lr,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": seed,
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "device": device,
    }

    # ============================================================
    # [EVAL MODE]  → 가장 최신 시간 폴더 자동 선택
    # ============================================================
    if is_eval:
        # 1) 먼저 root_dir 바로 아래에 policy_best.ckpt 가 있으면 그걸 사용
        ckpt_dir = ckpt_root_dir
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        if not os.path.exists(best_ckpt):
            # 2) 없으면, MMDD_HHMM 형식 하위 폴더 중 가장 최신 선택
            latest_sub = find_latest_timestamped_subdir(ckpt_root_dir)
            if latest_sub is None:
                raise FileNotFoundError(
                    f"[EVAL] No policy_best.ckpt in {ckpt_root_dir} "
                    f"and no timestamped subdirectories were found."
                )
            ckpt_dir = latest_sub
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"[EVAL] policy_best.ckpt not found in {ckpt_dir}")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[EVAL] dataset_stats.pkl not found in {ckpt_dir}")

        print(f"[EVAL] Using checkpoint dir: {ckpt_dir}")
        print(f"[INFO] Loading checkpoint from {best_ckpt}")

        policy = make_policy(policy_class, policy_config).to(device)

        ckpt = torch.load(best_ckpt, map_location=device)
        if policy_class == "ACT":
            # 학습에서 DETRVAE state_dict만 저장했으므로, model에만 로드
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            policy.model.load_state_dict(state_dict, strict=False)
        else:
            policy.load_state_dict(ckpt)

        policy.eval()

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        print(f"[INFO] Loaded dataset stats from {stats_path}")

        print("\n✅ Model ready for inference!")
        print("   (이 ckpt_path를 Isaac Sim / ROS2 infer 노드에서 사용하면 된다)\n")
        return

    # ============================================================
    # [TRAIN MODE]  → 실행 시간 기반으로 하위 폴더 생성 후 그 안에 저장
    # ============================================================
    # 예: /home/.../checkpoints/ur10e_swing/1116_1203/
    timestamp = datetime.now().strftime("%m%d_%H%M")
    ckpt_dir = os.path.join(ckpt_root_dir, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[TRAIN] Checkpoints will be saved under: {ckpt_dir}")

    config["ckpt_dir"] = ckpt_dir

    # 데이터 로드
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size, batch_size
    )

    # dataset 통계 저장
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {ckpt_dir}/dataset_stats.pkl")

    # 학습
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, best_ckpt_path)
    print(f"[INFO] Best ckpt saved -> {best_ckpt_path} (val_loss={min_val_loss:.6f})")


# -------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='run inference instead of training')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='루트 checkpoint 디렉터리')
    parser.add_argument('--policy_class', type=str, required=True)
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--kl_weight', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dim_feedforward', type=int)
    parser.add_argument('--temporal_agg', action='store_true')

    main(vars(parser.parse_args()))
