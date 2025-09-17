# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle: Joint-Hold (고정 target joint 유지) 환경 설정
"""

from __future__ import annotations
from isaaclab.utils import configclass

# Base EnvCfg
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg


@configclass
class PolishingPoseHoldEnvCfg(UR10eSpindleEnvCfg):
    """훈련/기본 구동용 joint-hold 환경"""

    def __post_init__(self):
        super().__post_init__()
        # 학습 안정화를 위한 기본 파라미터
        self.actions.arm_action.scale = 0.2
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0


@configclass
class PolishingPoseHoldEnvCfg_PLAY(PolishingPoseHoldEnvCfg):
    """시각화/디버깅 환경"""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
