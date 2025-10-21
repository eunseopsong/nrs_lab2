# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------
# Title: UR10e + Spindle (Environment with Contact Sensor)
# Author: Seungjun Song (NRS Lab)
# -----------------------------------------------------------------------------
"""
Manager-based Isaac Lab environment for the UR10e robot equipped with a spindle tool.

Key features:
- Horizon-based joint & position trajectory tracking
- Exponential-shaped reward for position tracking
- End-effector position observation (get_ee_pos)
- Contact/camera sensor integration (optional)
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
from isaaclab.sensors import ContactSensorCfg, CameraCfg

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

        # ✅ EE position (현재 엔드이펙터 위치)
        ee_pose = ObsTerm(
            func=local_obs.get_ee_pose,
            params={"asset_name": "robot", "frame_name": "wrist_3_link"},
        )

        # ✅ HDF5 기반 horizon-length trajectory 관측
        target_joints = ObsTerm(
            func=local_obs.get_hdf5_target_joints,
            params={"horizon": 5},
        )
        target_positions = ObsTerm(
            func=local_obs.get_hdf5_target_positions,
            params={"horizon": 5},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True  # ✅ 추가됨: dict → tensor 자동 병합

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
    # joint_tracking_reward = RewTerm(
    #     func=local_rewards.joint_tracking_reward,
    #     weight=1.0,
    # )

    # ✅ EE Position Tracking Reward
    position_tracking_reward = RewTerm(
        func=local_rewards.position_tracking_reward,
        weight=1.0,
    )

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
