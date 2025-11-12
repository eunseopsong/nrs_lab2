#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT training script (for offline UR10e dataset)
"""

import os
import pickle
import argparse
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from custom.custom_constants import DT, TASK_CONFIGS
from utils import load_data, compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy


def main(args):
    # ------------------------------------------------------------
    # 0. 기기 설정
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    # ------------------------------------------------------------
    # 1. CLI 파라미터
    # ------------------------------------------------------------
    task_name = args["task_name"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]
    seed = args["seed"]
    lr = args["lr"]
    dataset_dir_override = args.get("dataset_dir", None)

    # ------------------------------------------------------------
    # 2. Task 설정
    # ------------------------------------------------------------
    if not task_name in TASK_CONFIGS:
        raise KeyError(f"[ERROR] task_name '{task_name}' not found in TASK_CONFIGS.")

    task_config = TASK_CONFIGS[task_name]
    dataset_dir = dataset_dir_override or task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config.get("episode_len", 100)
    camera_names = task_config["camera_names"]

    print(f"[INFO] task_name      = {task_name}")
    print(f"[INFO] dataset_dir    = {dataset_dir}")
    print(f"[INFO] num_episodes   = {num_episodes}")
    print(f"[INFO] camera_names   = {camera_names}")

    # ------------------------------------------------------------
    # 3. 정책 설정
    # ------------------------------------------------------------
    state_dim = 6  # UR10e: 6 DOF
    lr_backbone = 1e-5
    backbone = "resnet18"

    if policy_class == "ACT":
        policy_config = {
            "lr": lr,
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

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": lr,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": seed,
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": False,
        "device": device,
    }

    # ------------------------------------------------------------
    # 4. 데이터 로드
    # ------------------------------------------------------------
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir, num_episodes, camera_names, batch_size, batch_size
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {ckpt_dir}/dataset_stats.pkl")

    # ------------------------------------------------------------
    # 5. 학습
    # ------------------------------------------------------------
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"[INFO] Best ckpt saved -> {ckpt_path} (val_loss={min_val_loss:.6f})")


# -------------------------------------------------------------------------
# 헬퍼 함수들
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
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    device = config["device"]

    set_seed(seed)
    policy = make_policy(policy_class, policy_config).to(device)
    optimizer = make_optimizer(policy_class, policy)

    train_history, validation_history = [], []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = [forward_pass(d, policy, device) for d in val_dataloader]
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        # Epoch summary
        start = (batch_idx + 1) * epoch
        end = (batch_idx + 1) * (epoch + 1)
        epoch_summary = compute_dict_mean(train_history[start:end])
        print(f"Train loss: {epoch_summary['loss']:.5f}")

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plt.figure()
        train_values = [x[key].item() for x in train_history]
        val_values = [x[key].item() for x in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_values)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(val_values)), val_values, label="val")
        plt.legend()
        plt.tight_layout()
        plt.title(key)
        plt.savefig(os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png"))
        plt.close()
    print(f"[INFO] Saved plots to {ckpt_dir}")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--policy_class", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--dataset_dir", type=str, required=False)
    parser.add_argument("--kl_weight", type=int, required=False)
    parser.add_argument("--chunk_size", type=int, required=False)
    parser.add_argument("--hidden_dim", type=int, required=False)
    parser.add_argument("--dim_feedforward", type=int, required=False)
    parser.add_argument("--temporal_agg", action="store_true")
    main(vars(parser.parse_args()))
