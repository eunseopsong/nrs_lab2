# SPDX-License-Identifier: BSD-3-Clause
"""
DirectRLEnv config for UR10e + spindle (matches the Cartpole direct template style).
- Uses your local USD robot prim at /World/ur10e_w_spindle_robot
- Action: 6-dim joint commands (Δq; scaling is handled in DirectRLEnv)
- Observation: 12-dim (q, dq)
"""

from typing import Sequence, Optional
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# 로봇 설정 (네가 쓰던 cfg를 그대로 사용)
from assets.assets.robots.ur10e_w_spindle import UR10E_W_SPINDLE_HIGH_PD_CFG

# --- 조인트 이름(USD의 실제 이름과 일치해야 함) ---
UR10E_JOINTS: Sequence[str] = [
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
    action_space = len(UR10E_JOINTS)           # 6
    observation_space = 2 * len(UR10E_JOINTS)  # q(6) + dq(6) = 12
    state_space = 0

    # ===== Simulation =====
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,   # 보통 decimation과 맞춤
        # substeps=1,
    )

    # ===== Robot =====
    robot_cfg: ArticulationCfg = UR10E_W_SPINDLE_HIGH_PD_CFG.replace(
        # prim_path="/World/ur10e_w_spindle_robot",  # 단일 env 기준
        prim_path="/World/Robot",  # 단일 env 기준
    )

    # ===== Scene =====
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,          # train.py에서 --num_envs로 오버라이드됨
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ===== 사용자 정의 스케일/이름 =====
    action_scale: float = 0.2           # Δq(rad) 스케일 (필요시 0.01~0.05로 줄여도 됨)
    joint_names: Sequence[str] = UR10E_JOINTS
    ee_frame_name: str = "wrist_3_link" # EE 기준 프레임명 (필요시 tool0/ee_link 등으로 변경)

    # ===== TCP 오프셋 (EE 로컬 프레임 기준; [x, y, z] in meters) =====
    # 0.185 m spindle 길이 (+Z 방향). 음수도 가능: (0, 0, -0.185) 등
    tcp_offset: Optional[Sequence[float]] = (0.0, 0.0, 0.185)

    # ===== 타깃 방문/보상 파라미터 (num_envs=1에 튜닝) =====
    reach_tol: float = 0.03           # [m] 도달 반경
    hold_steps: int = 5               # 연속 근접 유지 스텝
    max_phase_steps: int = 150        # 한 페이즈 최대 유지 스텝(정체 방지)

    # 보상 가중치
    w_pos: float = 10.0               # 위치 오차 L2 가중치
    w_ori: float = 6.0                # 자세 오차(툴 z축을 월드 -Z로) 가중치
    w_dq: float = 1e-3                # 관절 속도 패널티
    w_act: float = 5e-4               # 액션 크기 패널티

    # 보너스/패널티 (왕복 유도)
    bonus_reach: float = 1.0          # 목표 근접 보너스 (작게)
    bonus_phase: float = 10.0         # 페이즈 전환 보너스 (크게)
    phase_step_penalty: float = 0.01  # 페이즈 정체 패널티 계수

    # 홈포즈 유지 (옵션)
    enable_home_keep: bool = False
    w_home: float = 0.5
