# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle (manager-based): target joint 값 추종 환경
- Joint command error + tanh shaped reward
- Termination: trajectory 끝 or 시간 기반
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

# Reach MDP 유틸
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# ✅ 내 모듈을 확실히 importlib로 불러오기
import importlib
local_obs = importlib.import_module("nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.observations")
local_rewards = importlib.import_module("nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.rewards")

# 로봇 CFG
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
    load_hdf5 = EventTerm(
        func=local_obs.load_hdf5_trajectory,   # ✅ observations.py에서 가져오기
        mode="reset",
        params={
            "trajectory": None,
        },
    )



# ---------- Rewards ----------
@configclass
class RewardsCfg:
    joint_command_error = RewTerm(
        func=local_rewards.joint_command_error,
        weight=1.0,
    )
    joint_command_error_tanh = RewTerm(
        func=local_rewards.joint_command_error_tanh,
        weight=1.0,
        params={"std": 0.5},
    )


# ---------- Terminations ----------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # reached_end = DoneTerm(func=local_rewards.reached_end)


# ---------- EnvCfg ----------
@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=32, env_spacing=2.5)
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

        # 로봇 세팅
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # action 세팅
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
        )
