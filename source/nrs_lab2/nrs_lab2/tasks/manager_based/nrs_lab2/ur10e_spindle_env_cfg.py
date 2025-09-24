# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle (manager-based, BC policy 추종)
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
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from nrs_lab2.nrs_lab2.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2 import rewards as local_rewards

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

# ---------- Terminations ----------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

# ---------- EnvCfg ----------
@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=32, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 30.0

        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 🔥 Action = BC policy prediction
        def bc_action_wrapper(env, obs):
            q_current = obs[:, :6]
            q_pred = local_rewards.get_bc_action(env, obs)
            scale = 0.2
            return (q_pred - q_current) / scale

        self.actions.arm_action = ActionTermCfg(
            func=bc_action_wrapper,
            asset_name="robot",
        )
