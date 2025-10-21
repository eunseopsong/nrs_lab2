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

This environment tracks target joint trajectories and adds a Contact Sensor to the
end-effector to measure force / torque feedback.

Key features:
- Joint trajectory tracking with tanh-shaped reward.  
- Trajectory or time-based termination.  
- End-effector contact force sensor integration for feedback and RL observation.  
- Fully compatible with Isaac Lab’s manager-based MDP structure.

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
import math

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
from isaaclab.sensors import ContactSensorCfg, CameraCfg  # ✅ Camera sensor 추가

# Reach MDP utilities
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# Local modules (dynamically import for robustness)
local_obs = importlib.import_module(
    "nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations"
)
local_rewards = importlib.import_module(
    "nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.rewards"
)

# Robot and sensor assets
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

    # Contact sensor (replicated per env)
    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Robot/wrist_3_link",
    #     update_period=0.0,
    #     history_length=10,
    #     debug_vis=True,
    # )

    # -----------------------------------------------------------------------------
    # Camera Sensor Configuration
    # -----------------------------------------------------------------------------
    # 오일러 각(-180, 0, -180)을 쿼터니언(ros convention)으로 직접 계산한 값:
    # qx, qy, qz, qw = (0, 0, 0, 1)와 동일한 방향이지만 180° 회전 상태이므로 약간 다름.
    # 실제 변환 결과: (x=0, y=0, z=1, w=0) (즉, 180도 yaw)
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
    #         rot=(0.0, 0.0, 1.0, 0.0),  # ← ✅ 함수 대신 직접 쿼터니언 기입
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
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        actions = ObsTerm(func=mdp.last_action)

        # ee_state = ObsTerm(func=local_obs.get_ee_observation)   # ✅ 추가

        # HDF5 기반 target 예측 (trajectory 기반)
        target_future = ObsTerm(
            func=local_obs.get_hdf5_target_future,
            params={"horizon": 5},
        )

        # ✅ Contact Sensor 데이터 추가 (평균 Fx, Fy, Fz + dummy Tx,Ty,Tz)
        # contact_forces = ObsTerm(
        #     func=local_obs.get_contact_forces,
        #     params={"sensor_name": "contact_forces"},  # scene.contact_forces 이름과 동일해야 함
        # )

        # ✅ Camera distance 추가
        # camera_distance = ObsTerm(
        #     func=local_obs.get_camera_distance,
        #     params={"sensor_name": "camera"},
        # )

        # camera_normals = ObsTerm(
        #     func=local_obs.get_camera_normals,
        #     params={"sensor_name": "camera"},
        # )


        def __post_init__(self):
            # 관측 오염(노이즈) 허용
            self.enable_corruption = True
            # ✅ 모든 관측값을 하나의 벡터로 결합 (policy input으로 전달)
            self.concatenate_terms = True

    # RL 정책에서 사용할 observation 그룹
    policy: PolicyCfg = PolicyCfg()



# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------

@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    load_hdf5 = EventTerm(
        func=local_obs.load_hdf5_trajectory,
        mode="reset",
        params={"trajectory": None},
    )

# -----------------------------------------------------------------------------
# Rewards (Position-only, Exponential Lookahead)
# Version: v1
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    joint_tracking_reward = RewTerm(
        func=local_rewards.joint_tracking_reward,
        weight=1.0,
    )

    # (2) Contact stability reward
    # contact_force_reward = RewTerm(
    #     func=local_rewards.contact_force_reward,
    #     weight=0.05,
    #     params={
    #         "sensor_name": "contact_forces",
    #         "fz_min": 5.0,
    #         "fz_max": 20.0,
    #     },
    # )

    # # (3) ✅ Camera distance reward (스핀들 길이 유지)
    # camera_distance_reward = RewTerm(
    #     func=local_rewards.camera_distance_reward,
    #     weight=0.05,
    #     params={
    #         "target_distance": 0.185,  # spindle 길이
    #         "sigma": 0.035,             # 거리 허용 오차 범위 (±2cm)
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
