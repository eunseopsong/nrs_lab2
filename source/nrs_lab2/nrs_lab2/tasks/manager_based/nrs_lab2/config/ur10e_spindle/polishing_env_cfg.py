# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle: Pose-Hold (월드 기준 고정 포즈 유지) 설정.
- 타깃 (x,y,z,r,p,y) 및 리워드 가중치/스케일을 여기서 주입.
"""

from __future__ import annotations
import math
from isaaclab.utils import configclass

# !!! 중첩 패키지 경로 주의 !!!
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg
from nrs_lab2.nrs_lab2.robots.ur10e_w_spindle import EE_FRAME_NAME


def rpy_to_quat_wxyz(r: float, p: float, y: float):
    """(roll, pitch, yaw) -> (w,x,y,z)"""
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
    )


# ===== 원하는 포즈 (예시) =====
TARGET_POS_XYZ = (0.419164, -0.394628, 0.140684)  # [m]
TARGET_RPY_XYZ = (2.98492, -0.152798, -2.3346)    # [rad]
TARGET_QUAT_WXYZ = rpy_to_quat_wxyz(*TARGET_RPY_XYZ)


@configclass
class PolishingPoseHoldEnvCfg(UR10eSpindleEnvCfg):
    """훈련/기본 구동용 포즈-홀드 환경"""

    def __post_init__(self):
        super().__post_init__()

        # (1) 타깃 포즈 주입
        self.rewards.position_fixed.params["target_pos_xyz"] = TARGET_POS_XYZ
        self.rewards.position_fixed_tanh.params["target_pos_xyz"] = TARGET_POS_XYZ
        self.rewards.orientation_fixed.params["target_quat_wxyz"] = TARGET_QUAT_WXYZ

        # (2) 안전하게 EE 링크명 고정
        self.rewards.position_fixed.params["asset_cfg"].body_names = [EE_FRAME_NAME]
        self.rewards.position_fixed_tanh.params["asset_cfg"].body_names = [EE_FRAME_NAME]
        self.rewards.orientation_fixed.params["asset_cfg"].body_names = [EE_FRAME_NAME]

        # (3) 액션 스케일/시뮬 파라미터
        self.actions.arm_action.scale = 0.2
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0


@configclass
class PolishingPoseHoldEnvCfg_PLAY(PolishingPoseHoldEnvCfg):
    """시각화/디버깅"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
