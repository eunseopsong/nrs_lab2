# SPDX-License-Identifier: BSD-3-Clause
"""
Configuration for UR10e with spindle tool environment.
"""

from __future__ import annotations
import os
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, UsdFileCfg, GroundPlaneCfg
from isaaclab.managers import (
    ActionTermCfg as ActionTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    EventTermCfg as EventTerm,
    TerminationTermCfg as TermTerm,
)
from isaaclab.envs.mdp.actions.joint_actions import JointPositionActionCfg

# 👇 우리가 만든 함수들 import
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations import (
    get_hdf5_target,
    load_hdf5_trajectory,
)
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.rewards import (
    joint_command_error,
    joint_command_error_tanh,
    visualize_tracking,
)


# ------------------------------------------------------
# Scene configuration
# ------------------------------------------------------
@configclass
class SpindleSceneCfg:
    """Scene configuration for UR10e + Spindle"""

    # Robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path=os.path.expanduser("~/isaac/isaac_save/ur10e_w_spindle.usd"),
            visible=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": -1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": 1.57,
                "wrist_3_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(),
    )

    # Workpiece
    workpiece = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Workpiece",
        spawn=UsdFileCfg(
            usd_path=os.path.expanduser("~/isaac/isaac_save/concave_surface.usd"),
            visible=True,
        ),
    )


# ------------------------------------------------------
# Environment configuration
# ------------------------------------------------------
@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment config for UR10e + Spindle"""

    # Scene
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=32, env_spacing=2.5)

    # Actions
    actions = {
        "arm_action": ActionTerm(
            func=JointPositionActionCfg,
            params=dict(
                asset_name="robot",
                joint_names=[".*"],
                scale=0.2,
            ),
        )
    }

    # Observations
    observations = {
        "policy": dict(
            concatenate_terms=True,
            terms={
                "joint_pos": ObsTerm(func="joint_pos_rel"),
                "joint_vel": ObsTerm(func="joint_vel_rel"),
                "actions": ObsTerm(func="last_action"),
                "target_joint": ObsTerm(
                    func=get_hdf5_target,
                    params={"env_ids": lambda env: range(env.num_envs)},
                ),
            },
        )
    }

    # Rewards
    rewards = {
        "joint_command_error": RewTerm(func=joint_command_error, weight=1.0),
        "joint_command_error_tanh": RewTerm(func=joint_command_error_tanh, weight=1.0),
    }

    # Events
    events = {
        "reset_robot_joints": EventTerm(
            func="reset_joints_by_scale",
            params={"position_range": (0.75, 1.25), "velocity_range": (0.0, 0.0)},
            mode="reset",
        ),
        "load_hdf5": EventTerm(
            func=load_hdf5_trajectory,
            params={
                "file_path": os.path.expanduser("~/nrs_lab2/datasets/joint_recording.h5"),
                "dataset_key": "joint_positions",
            },
            mode="reset",
        ),
    }

    # Terminations
    terminations = {
        "time_out": TermTerm(func="time_out", time_out=True),
    }

    # Visualization hook
    extras = {
        "visualize_tracking": visualize_tracking,
    }
