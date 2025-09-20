# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle (manager-based): HDF5 trajectory tracking 환경
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
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp import rewards as local_rewards
from nrs_lab2.nrs_lab2.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG


# ---------- Scene ----------
@configclass
class SpindleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: AssetBaseCfg = MISSING


# ---------- Actions ----------
@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING


# ---------- Observations ----------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------- Events ----------
@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.75, 1.25), "velocity_range": (0.0, 0.0)},
    )

    # HDF5 trajectory 로드
    load_hdf5 = EventTerm(
        func=local_rewards.load_hdf5_trajectory,
        mode="reset",
    )


# ---------- Rewards ----------
@configclass
class RewardsCfg:
    joint_target_error = RewTerm(
        func=local_rewards.joint_target_error,
        weight=-1.0,
    )
    joint_target_tanh = RewTerm(
        func=local_rewards.joint_target_tanh,
        weight=1.0,
    )


# ---------- Terminations ----------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------- EnvCfg ----------
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
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 60.0

        # 로봇 주입
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 액션: 모든 조인트 position 제어
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
        )
