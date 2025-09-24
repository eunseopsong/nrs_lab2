# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle: Joint-Hold 환경 (BC imitation)
"""

from __future__ import annotations
from isaaclab.utils import configclass

from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp import rewards as local_rewards
from isaaclab.managers import EventTermCfg as EventTerm


@configclass
class PolishingPoseHoldEnvCfg(UR10eSpindleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = 0.2
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0

        # ✅ load_bc_policy 호출로 교체
        self.events.load_bc = EventTerm(
            func=local_rewards.load_bc_policy,
            mode="reset",
            params={
                "file_path": "/home/eunseop/nrs_lab2/datasets/bc_policy.pth",
            },
        )


@configclass
class PolishingPoseHoldEnvCfg_PLAY(PolishingPoseHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
