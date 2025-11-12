#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT training / (미구현) offline eval 스크립트
환경에 맞게 정리한 버전
"""

import os
import time
import math
import pickle
import argparse
from copy import deepcopy

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange  # 아직 eval 쪽에서만 쓰지만 일단 둠

from custom.custom_constants import DT, TASK_CONFIGS
from utils import load_data, sample_box_pose, sample_insertion_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy

# Isaac sim / mujoco env를 쓰지 않는 버전이라 sim_env 관련은 사용 안 함


def main(args):
    set_seed(1)

    # 0) 기기 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    # 1) CLI 파라미터
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    dataset_dir_override = args.get("dataset_dir", None)

    # 2) 태스크 설정 읽기
    is_sim = task_name.startswith("sim_")
    if is_sim:
        # 네 환경에선 안 쓸 거라 뺐지만, 남겨둠
        raise NotImplementedError("sim_ 태스크는 이 축약 버전에서 다루지 않았습니다.")
    else:
        task_config = TASK_CONFIGS[task_name]

    # 기본값은 custom_constants 에서
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # 만약 CLI에서 dataset_dir 직접 준 경우 그걸 최우선으로 사용
    if dataset_dir_override is not None:
        dataset_dir = dataset_dir_override
        print(f"[INFO] dataset_dir overridden from CLI: {dataset_dir}")

    print(f"[INFO] task_name      = {task_name}")
    print(f"[INFO] dataset_dir    = {dataset_dir}")
    print(f"[INFO] num_episodes   = {num_episodes}")
    print(f"[INFO] camera_names   = {camera_names}")

    # 3) 고정/모델 설정
    # 네 hdf5는 6 joint만 들어있으니까 6으로
    state_dim = 6

    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        # 이번 버전에선 항상 offline dataset 학습이므로 False로 둬도 상관없음
        "real_robot": False,
        "device": device,
    }

    # 4) eval 모드면 지금은 안 함
    if is_eval:
        raise NotImplementedError(
            "현재 버전에서는 --eval 경로를 실제 로봇/시뮬과 연결하지 않았습니다. "
            "먼저 학습만 돌리고 싶으면 --eval 빼고 실행하세요."
        )

    # 5) 데이터 로드 (episode_*.hdf5 가 있는 디렉터리여야 함)
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
    )

    # 6) 저장 디렉터리
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {stats_path}")

    # 7) 학습
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # 8) best ckpt 저장
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"[INFO] Best ckpt, val loss {min_val_loss:.6f} @ epoch {best_epoch}")
    print(f"[INFO] saved -> {ckpt_path}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    # 두 정책 다 내부에서 optimizer 만들어서 넘기게 돼 있음
    return policy.configure_optimizers()


def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    device = config["device"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config).to(device)
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        # ---------- validation ----------
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for _, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        print(" ".join(f"{k}: {v.item():.3f}" for k, v in epoch_summary.items()))

        # ---------- training ----------
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        # 이번 epoch 의 train 평균 뽑기
        # (batch_idx+1)개가 이번 epoch에서 추가됐다고 보고 슬라이스
        start = (batch_idx + 1) * epoch
        end = (batch_idx + 1) * (epoch + 1)
        epoch_summary = compute_dict_mean(train_history[start:end])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        print(" ".join(f"{k}: {v.item():.3f}" for k, v in epoch_summary.items()))

        # 중간 체크포인트
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # 마지막 모델도 저장
    last_ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_last_seed_{seed}.ckpt")
    torch.save(policy.state_dict(), last_ckpt_path)

    # 베스트도 따로 저장
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    best_ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, best_ckpt_path)
    print(
        f"Training finished: seed {seed}, best val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # 곡선 저장
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # train_history[0] 에 있는 키들 기준으로 그림 그림
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f"[INFO] Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint dir")
    parser.add_argument("--policy_class", type=str, required=True, help="ACT or CNNMLP")
    parser.add_argument("--task_name", type=str, required=True, help="task_name in custom_constants")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    # 선택: 데이터셋 디렉터리를 강제로 지정하고 싶은 경우
    parser.add_argument("--dataset_dir", type=str, required=False, help="(optional) override dataset dir")

    # ACT 전용
    parser.add_argument("--kl_weight", type=int, required=False)
    parser.add_argument("--chunk_size", type=int, required=False)
    parser.add_argument("--hidden_dim", type=int, required=False)
    parser.add_argument("--dim_feedforward", type=int, required=False)
    parser.add_argument("--temporal_agg", action="store_true")

    main(vars(parser.parse_args()))
