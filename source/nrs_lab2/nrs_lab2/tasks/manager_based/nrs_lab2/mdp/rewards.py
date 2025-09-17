# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState  # target joint q1~q6 퍼블리시한다고 가정

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_error_magnitude,
    quat_mul,
    quat_inv,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================
# ROS2 Subscriber Node (target joints q1~q6)
# ============================================================
class TargetJointSubscriber(Node):
    def __init__(self):
        super().__init__("target_joint_subscriber")
        self.target_joints = torch.zeros(6, dtype=torch.float32)
        self.sub = self.create_subscription(
            JointState,
            "/target_joints",  # C++ node에서 퍼블리시하는 토픽 이름
            self.joint_callback,
            10,
        )

    def joint_callback(self, msg: JointState):
        # msg.position = [q1, q2, q3, q4, q5, q6]
        if len(msg.position) >= 6:
            self.target_joints = torch.tensor(msg.position[:6], dtype=torch.float32)


# ============================================================
# ROS2 초기화 및 글로벌 subscriber 객체
# ============================================================
rclpy.init(args=None)
_target_joint_node = TargetJointSubscriber()


def _update_ros2_once():
    """Spin ROS2 node non-blocking (한번씩만 돌려주기)."""
    rclpy.spin_once(_target_joint_node, timeout_sec=0.0)


# ============================================================
# Reward Functions
# ============================================================
def joint_position_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    현재 로봇 조인트 상태(q1~q6)와 ROS2에서 받은 target joint 상태 차이 계산.
    """
    _update_ros2_once()  # 최신 값 갱신

    # 현재 로봇 조인트 값 (batch x 6)
    cur_q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids, 0]

    # target joints (6,)
    target_q = _target_joint_node.target_joints.to(cur_q.device)

    # shape 맞춰주기 (batch x 6)
    target_q = target_q.unsqueeze(0).repeat(cur_q.shape[0], 1)

    # L2 norm error
    error = torch.norm(cur_q - target_q, dim=-1)
    return error


def joint_position_tanh_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.1) -> torch.Tensor:
    """
    조인트 위치 오차 기반 tanh-shaped reward.
    """
    error = joint_position_error(env, asset_cfg)
    return torch.tanh(-error / std)
