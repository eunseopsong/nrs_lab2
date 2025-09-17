# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import torch
import os

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------------------
# A) 명령(커맨드) 기반 리워드 — 관측 파이프라인 호환 목적 (필요시 그대로 사용)
# --------------------------------------------------------------------------------------
def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    des_pos_b = cmd[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    des_pos_b = cmd[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    dist = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1.0 - torch.tanh(dist / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    des_quat_b = cmd[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)


# --------------------------------------------------------------------------------------
# B) 고정 타깃 포즈 리워드 — (x,y,z,r,p,y) 유지
# --------------------------------------------------------------------------------------
def _target_pos(env: ManagerBasedRLEnv, xyz: Tuple[float, float, float]) -> torch.Tensor:
    return torch.tensor(xyz, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

def _target_quat(env: ManagerBasedRLEnv, wxyz: Tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor(wxyz, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

def position_fixed_error(env: ManagerBasedRLEnv, target_pos_xyz: Tuple[float, float, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    p_t = _target_pos(env, target_pos_xyz)
    return torch.norm(p - p_t, dim=1)

def position_fixed_tanh(env: ManagerBasedRLEnv, std: float, target_pos_xyz: Tuple[float, float, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    p_t = _target_pos(env, target_pos_xyz)
    d = torch.norm(p - p_t, dim=1)
    return 1.0 - torch.tanh(d / std)

def orientation_fixed_error(env: ManagerBasedRLEnv, target_quat_wxyz: Tuple[float, float, float, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    q_t = _target_quat(env, target_quat_wxyz)
    return quat_error_magnitude(q, q_t)

def small_error_bonus_fixed(
    env: ManagerBasedRLEnv,
    pos_tol: float,
    ang_tol: float,
    target_pos_xyz: Tuple[float, float, float],
    target_quat_wxyz: Tuple[float, float, float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    q = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    p_t = _target_pos(env, target_pos_xyz)
    q_t = _target_quat(env, target_quat_wxyz)
    pos_err = torch.norm(p - p_t, dim=1)
    ang_err = quat_error_magnitude(q, q_t)
    ok = (pos_err < pos_tol) & (ang_err < ang_tol)
    return ok.float()


# --------------------------------------------------------------------------------------
# C) 고정된 타깃 조인트 리워드 — target_joints.txt 로드
# --------------------------------------------------------------------------------------
TARGET_FILE = os.path.expanduser("~/nrs_lab2/datasets/target_joints.txt")

def _load_target_joints(num_joints: int = 6, device: str = "cpu") -> torch.Tensor:
    if not os.path.exists(TARGET_FILE):
        return torch.zeros(num_joints, dtype=torch.float32, device=device)
    try:
        with open(TARGET_FILE, "r") as f:
            line = f.readline().strip()
        values = [float(x) for x in line.split()]
        if len(values) != num_joints:
            return torch.zeros(num_joints, dtype=torch.float32, device=device)
        return torch.tensor(values, dtype=torch.float32, device=device)
    except Exception:
        return torch.zeros(num_joints, dtype=torch.float32, device=device)

def joint_target_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    joint_pos = env.obs_buf[:, env.cfg.obs_group.joint_pos_ids]
    target = _load_target_joints(joint_pos.shape[1], device=env.device)
    target = target.unsqueeze(0).expand_as(joint_pos)
    return torch.mean((joint_pos - target) ** 2, dim=-1)

def joint_target_tanh(env: ManagerBasedRLEnv) -> torch.Tensor:
    joint_pos = env.obs_buf[:, env.cfg.obs_group.joint_pos_ids]
    target = _load_target_joints(joint_pos.shape[1], device=env.device)
    target = target.unsqueeze(0).expand_as(joint_pos)
    error = torch.mean((joint_pos - target) ** 2, dim=-1)
    return 1.0 - torch.tanh(error)
