# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Title: UR10e + Spindle (Environment with Contact Sensor)
# Author: Seungjun Song (NRS Lab)
# -----------------------------------------------------------------------------
"""
Manager-based Isaac Lab environment for the UR10e robot equipped with a spindle tool.

This environment tracks target joint and position trajectories from HDF5 datasets,
and supports additional sensors such as contact and camera modules.

Key features:
- Horizon-based joint & position trajectory tracking
- Tanh/exponential-shaped reward formulation
- End-effector contact force & camera integration (optional)
- Compatible with Isaac Lab’s manager-based MDP structure

Example usage:
.. code-block:: bash

    ./isaaclab.sh -p nrs_lab2/tasks/manager_based/ur10e_spindle_env_cfg.py --enable_cameras
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import MISSING
import importlib
import torch
import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, CameraCfg  # ✅ 카메라 유지

# Reach manipulation utilities
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# -----------------------------------------------------------------------------
# Local modules (dynamic import)
# -----------------------------------------------------------------------------
local_obs = importlib.import_module(
    "nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations"
)
local_rewards = importlib.import_module(
    "nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.rewards"
)

# -----------------------------------------------------------------------------
# Robot asset
# -----------------------------------------------------------------------------
from assets.assets.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG

# -----------------------------------------------------------------------------
# Scene Configuration
# -----------------------------------------------------------------------------
@configclass
class SpindleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: AssetBaseCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/eunseop/isaac/isaac_save/concave_surface.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ✅ Contact Sensor (optional)
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Robot/wrist_3_link",
    #     update_period=0.0,
    #     history_length=10,
    #     debug_vis=True,
    # )

    # ✅ Camera Sensor (optional)
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Robot/wrist_3_link/camera_sensors",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane", "normals"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=18.14756,
    #         focus_distance=40.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.1, 1.0e5),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, -0.1, 0.0),
    #         rot=(0.0, 0.0, 1.0, 0.0),
    #         convention="ros",
    #     ),
    # )

# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING

# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 기본 joint 관측
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)

        # ✅ HDF5 기반 horizon-length trajectory 관측
        target_joints = ObsTerm(
            func=local_obs.get_hdf5_target_joints,
            params={"horizon": 5},
        )
        target_positions = ObsTerm(
            func=local_obs.get_hdf5_target_positions,
            params={"horizon": 5},
        )

        # ✅ 추가 센서 (필요 시 주석 해제)
        # contact_forces = ObsTerm(
        #     func=local_obs.get_contact_forces,
        #     params={"sensor_name": "contact_forces"},
        # )
        # camera_distance = ObsTerm(
        #     func=local_obs.get_camera_distance,
        #     params={"sensor_name": "camera"},
        # )
        # camera_normals = ObsTerm(
        #     func=local_obs.get_camera_normals,
        #     params={"sensor_name": "camera"},
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True  # 모든 항목을 단일 observation vector로 결합

    policy: PolicyCfg = PolicyCfg()

# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------
@configclass
class EventCfg:
    """Episode 시작 시 두 개의 trajectory (joints / positions)를 로드"""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # ✅ Joint trajectory 로드
    load_hdf5_joints = EventTerm(
        func=local_obs.load_hdf5_joints,
        mode="reset",
        params={
            "file_path": "/home/eunseop/nrs_lab2/datasets/joint_recording_filtered.h5",
            "dataset_key": "target_joints",
        },
    )

    # ✅ Position trajectory 로드
    load_hdf5_positions = EventTerm(
        func=local_obs.load_hdf5_positions,
        mode="reset",
        params={
            "file_path": "/home/eunseop/nrs_lab2/datasets/hand_g_recording.h5",
            "dataset_key": "target_positions",
        },
    )

# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    joint_tracking_reward = RewTerm(
        func=local_rewards.joint_tracking_reward,
        weight=1.0,
    )

    # ✅ 센서 기반 보상 (옵션)
    # contact_force_reward = RewTerm(
    #     func=local_rewards.contact_force_reward,
    #     weight=0.05,
    #     params={
    #         "sensor_name": "contact_forces",
    #         "fz_min": 5.0,
    #         "fz_max": 20.0,
    #     },
    # )
    # camera_distance_reward = RewTerm(
    #     func=local_rewards.camera_distance_reward,
    #     weight=0.05,
    #     params={
    #         "target_distance": 0.185,
    #         "sigma": 0.035,
    #     },
    # )

# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 30.0

        # Robot configuration
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Action configuration
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.2,
            use_default_offset=True,
        )
