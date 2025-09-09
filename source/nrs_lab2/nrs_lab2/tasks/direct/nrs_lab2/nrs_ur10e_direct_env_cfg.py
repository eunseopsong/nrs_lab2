# SPDX-License-Identifier: BSD-3-Clause
"""
DirectRLEnv config for UR10e + spindle (matches the Cartpole direct template style).
- Uses your local USD robot prim at /World/ur10e_w_spindle_robot
- Action: 6-dim joint commands (e.g., delta-position or torque hook는 DirectRLEnv에서 구현)
- Observation: 12-dim (q, dq) as a simple placeholder
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# 로봇 설정
from nrs_lab2.robots.ur10e_w_spindle import UR10E_W_SPINDLE_CFG
from nrs_lab2.robots.ur10e_w_spindle import UR10E_W_SPINDLE_HIGH_PD_CFG



# --- 조인트 이름(USD의 실제 이름과 일치해야 함) ---
UR10E_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

@configclass
class NrsUR10eDirectEnvCfg(DirectRLEnvCfg):
    # ===== Env 기본 파라미터 (Cartpole 템플릿과 동일한 구조) =====
    decimation = 2
    episode_length_s = 8.0

    # --- spaces 정의 ---
    # 6개 조인트 제어(예: Δq 명령)를 가정
    action_space = len(UR10E_JOINTS)               # 6
    # 관찰: q(6) + dq(6) = 12 (원하면 이후에 바꿔도 됨)
    observation_space = 2 * len(UR10E_JOINTS)      # 12
    state_space = 0

    # ===== Simulation =====
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        # substeps=1,
    )

    # ===== Robot(s) =====
    # 여러 env 복제 시 정규식 prim 경로를 쓰는 Cartpole 템플릿 스타일
    # robot_cfg: ArticulationCfg = UR10E_W_SPINDLE_CFG.replace(
    robot_cfg: ArticulationCfg = UR10E_W_SPINDLE_HIGH_PD_CFG.replace(
        # 단일 env면 정확 경로, 다중 env면 정규식 사용
        # 단일: prim_path="/World/ur10e_w_spindle_robot",
        # prim_path="/World/envs/env_.*/ur10e_w_spindle_robot",
        prim_path="/World/ur10e_w_spindle_robot",
        # prim_path="/World/envs/env_.*/ur10e_w_spindle_robot",
    )

    # ===== Scene =====
    # 우선 1~64에서 시작하고 VRAM/속도에 따라 늘려가
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,            # 필요 시 1 또는 256/512 등으로 조정
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ===== 사용자 정의 스케일/이름 =====
    # 액션 스케일: Δq(rad) 또는 토크/노멀라이즈를 DirectRLEnv 쪽 로직에서 해석
    # action_scale = 0.05
    action_scale = 0.2
    joint_names = UR10E_JOINTS
    ee_frame_name = "wrist_3_link"
