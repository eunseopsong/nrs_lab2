# SPDX-License-Identifier: BSD-3-Clause
"""
UR10e + Spindle (manager-based): 고정 (x, y, z, r, p, y) 포즈 유지용 베이스 EnvCfg
- 커맨드(Commands) 미사용: 관측에서 generated_commands('ee_pose') 제거
- 리워드로만 고정 타깃 포즈를 추종
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
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 표준 reach MDP 유틸(액션/관측/이벤트에 사용)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# 고정 포즈 리워드(로컬)
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp import rewards as local_rewards
# 로봇 CFG (중첩 패키지 경로 주의)
from nrs_lab2.nrs_lab2.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG, EE_FRAME_NAME


# ---------- Scene ----------
@configclass
class SpindleSceneCfg(InteractiveSceneCfg):
    """간단한 장면: Ground + Robot"""
    # ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    # 로봇(아래 EnvCfg에서 주입)
    robot: AssetBaseCfg = MISSING
    # 라이트
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


# ---------- Actions ----------
@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


# ---------- Observations (Commands 미사용) ----------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 기본 조인트 상태 + 마지막 액션
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


# ---------- Rewards (고정 포즈 추종) ----------
@configclass
class RewardsCfg:
    # 위치 오차 L2 페널티
    position_fixed = RewTerm(
        func=local_rewards.position_fixed_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "target_pos_xyz": MISSING,   # 상위 cfg에서 지정
        },
    )
    # 근접 보상(tanh)
    position_fixed_tanh = RewTerm(
        func=local_rewards.position_fixed_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": 0.1,
            "target_pos_xyz": MISSING,   # 상위 cfg에서 지정
        },
    )
    # 자세 오차(최단각) 페널티
    orientation_fixed = RewTerm(
        func=local_rewards.orientation_fixed_error,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "target_quat_wxyz": MISSING,  # 상위 cfg에서 지정
        },
    )

    # 안정화 패널티
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel   = RewTerm(func=mdp.joint_vel_l2,  weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})


# ---------- Terminations ----------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


# ---------- Curriculum (선택) ----------
@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500})
    joint_vel   = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "joint_vel",   "weight": -0.001, "num_steps": 4500})


# ---------- EnvCfg ----------
@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e(+spindle) 고정 포즈 유지용 베이스 환경"""

    # Scene
    scene: SpindleSceneCfg = SpindleSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 기본 시뮬 파라미터
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0

        # 로봇 주입 (로컬 USD 사용)
        self.scene.robot = UR10E_W_SPINDLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # EE 프레임명
        tip = EE_FRAME_NAME or "wrist_3_link"

        # 액션: 조인트 전부 position 명령
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True
        )

        # 리워드에서 EE 링크 지정 (타깃 포즈/쿼터니언은 상위 cfg에서 주입)
        self.rewards.position_fixed.params["asset_cfg"].body_names = [tip]
        self.rewards.position_fixed_tanh.params["asset_cfg"].body_names = [tip]
        self.rewards.orientation_fixed.params["asset_cfg"].body_names = [tip]
