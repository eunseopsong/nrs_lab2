# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import torch

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
