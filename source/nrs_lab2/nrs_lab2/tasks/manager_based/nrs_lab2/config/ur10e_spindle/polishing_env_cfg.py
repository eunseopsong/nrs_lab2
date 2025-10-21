# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle: Joint-Hold 환경 설정 (두 개의 trajectory 로드)
- Uses joint_recording_filtered.h5 (joints)
- Uses hand_g_recording.h5 (positions)
"""

from __future__ import annotations
from isaaclab.utils import configclass
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp import observations as local_obs
from isaaclab.managers import EventTermCfg as EventTerm


@configclass
class PolishingPoseHoldEnvCfg(UR10eSpindleEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 🔹 Simulation 설정
        self.sim.dt = 1.0 / 60.0
        self.sim.physics_dt = 1.0 / 60.0
        self.sim.substeps = 1
        self.sim.use_gpu_pipeline = True

        # 🔹 환경 설정
        self.actions.arm_action.scale = 0.2
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0

        # 🔹 HDF5 trajectory (joints)
        self.events.load_hdf5_joints = EventTerm(
            func=local_obs.load_hdf5_joints,
            mode="reset",
            params={
                "file_path": "/home/eunseop/nrs_lab2/datasets/joint_recording_filtered.h5",
                "dataset_key": "target_joints",
            },
        )

        # 🔹 HDF5 trajectory (positions)
        self.events.load_hdf5_positions = EventTerm(
            func=local_obs.load_hdf5_positions,
            mode="reset",
            params={
                "file_path": "/home/eunseop/nrs_lab2/datasets/hand_g_recording.h5",
                "dataset_key": "target_positions",
            },
        )


@configclass
class PolishingPoseHoldEnvCfg_PLAY(PolishingPoseHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
