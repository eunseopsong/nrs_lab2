# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle (manager-based): 포즈 유지용 베이스 EnvCfg
- 로봇 cfg는 assets/robots/ur10e_w_spindle.py 에 정의된 UR10E_W_SPINDLE_CFG 사용
"""

from __future__ import annotations
from dataclasses import MISSING

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
    SceneEntityCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp import rewards as local_rewards

# --- 여기서 로봇 cfg 불러옴 ---
from nrs_lab2.nrs_lab2.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG, EE_FRAME_NAME



@configclass
class UR10eSpindleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING


@configclass
class RewardsCfg:
    position_fixed = RewTerm(
        func=local_rewards.position_fixed_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_FRAME_NAME]), "target_pos_xyz": MISSING},
    )
    position_fixed_tanh = RewTerm(
        func=local_rewards.position_fixed_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_FRAME_NAME]), "std": 0.1, "target_pos_xyz": MISSING},
    )
    orientation_fixed = RewTerm(
        func=local_rewards.orientation_fixed_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[EE_FRAME_NAME]), "target_quat_wxyz": MISSING},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.75, 1.25), "velocity_range": (0.0, 0.0)},
    )


@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    scene: UR10eSpindleSceneCfg = UR10eSpindleSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
        )
