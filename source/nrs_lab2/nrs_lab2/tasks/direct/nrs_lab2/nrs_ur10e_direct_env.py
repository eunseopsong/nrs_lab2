# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from collections.abc import Sequence
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .nrs_ur10e_direct_env_cfg import NrsUR10eDirectEnvCfg


class NrsUR10eDirectEnv(DirectRLEnv):
    """UR10e 직접 RL 환경 (home pose 유지 + 두 타깃을 번갈아 방문).

    - Actions: Δq (rad), [-1,1] 클리핑 후 cfg.action_scale 배로 확대하여 position target에 적층
    - Observations: [q(6), dq(6)]
    - Rewards:
        * 기본: home pose 추종(선택) + 속도/액션 패널티(작게)
        * 확장: EE가 (0.5, -0.4, 0.1)와 (0.5,  0.4, 0.1)을 번갈아 방문하도록 위치 오차 패널티 및 도달 보너스
          - reach_tol 안에 연속 hold_steps 프레임 머물면 A<->B 페이즈 토글
          - max_phase_steps 초과 시 강제 토글(정체 방지)
    - Dones: timeout only
    """
    cfg: NrsUR10eDirectEnvCfg

    # ---------------- Init ----------------
    def __init__(self, cfg: NrsUR10eDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)  # _setup_scene() 내부에서 self.robot 생성

        # --- Joint indices (robust) ---
        self._jname = list(self.cfg.joint_names)
        jid_raw, _ = self.robot.find_joints(self._jname)
        if isinstance(jid_raw, (list, tuple)):
            flat = []
            for it in jid_raw:
                if isinstance(it, (list, tuple)):
                    flat.extend(it)
                else:
                    flat.append(int(it))
            idx_list = flat
        elif torch.is_tensor(jid_raw):
            idx_list = jid_raw.flatten().tolist()
        else:
            idx_list = [int(jid_raw)]
        assert len(idx_list) == len(self._jname), \
            f"[UR10e] Joint index count mismatch: names={self._jname} -> indices={idx_list}"
        self._jid = torch.as_tensor(idx_list, device=self.device, dtype=torch.long)  # (6,)

        # --- EE link index (robust) ---
        # cfg에 ee_link_name이 있으면 사용, 없으면 일반적인 후보를 탐색
        ee_candidates = []
        if hasattr(self.cfg, "ee_link_name") and self.cfg.ee_link_name:
            ee_candidates.append(self.cfg.ee_link_name)
        ee_candidates += ["ee_link", "tool0", "flange", "wrist_3_link"]
        ee_found = False
        for name in ee_candidates:
            try:
                bid, _ = self.robot.find_bodies([name])
                if isinstance(bid, (list, tuple)) and len(bid) > 0:
                    self._ee_id = int(bid[0] if not isinstance(bid[0], (list, tuple)) else bid[0][0])
                    ee_found = True
                    self._ee_name = name
                    break
                elif torch.is_tensor(bid):
                    self._ee_id = int(bid.flatten()[0].item())
                    ee_found = True
                    self._ee_name = name
                    break
            except Exception:
                pass
        if not ee_found:
            raise RuntimeError("EE 링크를 찾을 수 없습니다. cfg.ee_link_name을 지정하거나 링크명을 확인하세요.")

        # --- State refs ---
        self.q = self.robot.data.joint_pos   # (num_envs, dof)
        self.dq = self.robot.data.joint_vel  # (num_envs, dof)

        # Home pose (옵션): cfg.init_state.joint_pos에서 읽음
        q_ref_list = [self.robot.cfg.init_state.joint_pos[name] for name in self._jname]
        self.q_ref = torch.tensor(q_ref_list, device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,6)

        # --- Target positions (world frame) ---
        self.target_A = torch.tensor([0.5, -0.4, 0.1], device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,3)
        self.target_B = torch.tensor([0.5,  0.4, 0.1], device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,3)

        # --- Phase buffers ---
        self._phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 0->A, 1->B
        self._near_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._phase_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # --- Hyper-parameters (필요 시 cfg로 옮겨 사용 가능) ---
        self.reach_tol = getattr(self.cfg, "reach_tol", 0.03)          # [m] 목표 근접 반경
        self.hold_steps = getattr(self.cfg, "hold_steps", 10)          # 연속 유지 스텝 수
        self.max_phase_steps = getattr(self.cfg, "max_phase_steps", 600)  # 페이즈 최대 지속 스텝
        self.w_pos = getattr(self.cfg, "w_pos", 10.0)                  # 위치 오차 가중치
        self.w_dq = getattr(self.cfg, "w_dq", 1e-3)                    # 속도 패널티
        self.w_act = getattr(self.cfg, "w_act", 5e-4)                  # 액션 패널티
        self.bonus_reach = getattr(self.cfg, "bonus_reach", 2.0)       # 도달 보너스
        self.enable_home_keep = getattr(self.cfg, "enable_home_keep", False)
        self.w_home = getattr(self.cfg, "w_home", 0.5)                 # 홈포즈 유지(선택)

        # --- Safe default action ---
        self.actions = torch.zeros((self.num_envs, len(self._jname)), device=self.device, dtype=self.q.dtype)

    # ---------------- Scene ----------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        dl = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        dl.func("/World/Light", dl)

    # ---------------- Actions ----------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        dq_cmd = self.actions * self.cfg.action_scale
        q_now = self.q[:, self._jid]
        q_tgt = q_now + dq_cmd
        self.robot.set_joint_position_target(q_tgt, joint_ids=self._jid)

    # ---------------- Observations ----------------
    def _get_observations(self) -> dict:
        obs = torch.cat((self.q[:, self._jid], self.dq[:, self._jid]), dim=-1)  # (num_envs, 12)
        return {"policy": obs}

    # ---------------- Rewards ----------------
    def _get_rewards(self) -> torch.Tensor:
        # 1) EE 위치 얻기
        ee_pos = self._get_ee_pos()  # (num_envs, 3)

        # 2) 현재 페이즈별 타깃 선택
        tgt = torch.where(self._phase.unsqueeze(-1) == 0, self.target_A, self.target_B)  # (num_envs, 3)
        # (broadcast) target_A/B는 (1,3)이므로 자동 확장

        # 3) 위치 오차 및 근접 판정
        pos_err = ee_pos - tgt
        pos_err_sq = torch.sum(pos_err * pos_err, dim=-1)  # (num_envs,)
        near = pos_err_sq <= (self.reach_tol * self.reach_tol)

        # 4) 보상 계산
        rew = -self.w_pos * pos_err_sq \
              - self.w_dq * torch.sum(self.dq[:, self._jid] * self.dq[:, self._jid], dim=-1) \
              - self.w_act * torch.sum(self.actions * self.actions, dim=-1)

        if self.enable_home_keep:
            err_q = self.q[:, self._jid] - self.q_ref
            rew = rew - self.w_home * torch.sum(err_q * err_q, dim=-1)

        # 도달 보너스 (근접 상태에서만)
        rew = rew + self.bonus_reach * near.to(rew.dtype)

        # 5) 페이즈 토글 로직 업데이트
        self._update_phase(near)

        return rew

    # ---------------- Dones ----------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out = torch.zeros_like(time_out, dtype=torch.bool)
        return out, time_out

    # ---------------- Reset ----------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        q0 = self.robot.data.default_joint_pos[env_ids]
        dq0 = self.robot.data.default_joint_vel[env_ids]
        root = self.robot.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(q0, dq0, None, env_ids)

        # 페이즈/카운터 초기화
        self._phase[env_ids] = 0
        self._near_count[env_ids] = 0
        self._phase_step[env_ids] = 0

    # ---------------- Helpers ----------------
    def _get_ee_pos(self) -> torch.Tensor:
        """엔드이펙터(world) 위치 반환. Isaac Lab 버전별 데이터 레이아웃 차이를 대비해 여러 경로 시도."""
        d = self.robot.data
        # 1) 가장 흔한 경로: body_pos_w (num_envs, n_bodies, 3)
        if hasattr(d, "body_pos_w"):
            return d.body_pos_w[:, self._ee_id, :]
        # 2) link_pos_w
        if hasattr(d, "link_pos_w"):
            return d.link_pos_w[:, self._ee_id, :]
        # 3) body_state_w: (..., 13): pos(3)+quat(4)+lin vel(3)+ang vel(3)
        if hasattr(d, "body_state_w"):
            return d.body_state_w[:, self._ee_id, 0:3]
        # 4) link_state_w
        if hasattr(d, "link_state_w"):
            return d.link_state_w[:, self._ee_id, 0:3]
        raise RuntimeError("EE 위치 데이터를 찾을 수 없습니다. Isaac Lab 데이터 필드를 확인하세요.")

    def _update_phase(self, near: torch.Tensor) -> None:
        """목표 근접 연속성/최대 지속시간을 기준으로 A<->B 페이즈를 토글."""
        # 연속 근접 카운트 업데이트
        self._near_count = torch.where(near, self._near_count + 1, torch.zeros_like(self._near_count))
        # 페이즈 경과 스텝
        self._phase_step = self._phase_step + 1

        # 토글 조건 1: 연속 hold_steps 근접 유지
        toggle_hit = self._near_count >= self.hold_steps
        # 토글 조건 2: 페이즈가 너무 오래 지속될 때 강제 토글(정체 회피)
        toggle_timeout = self._phase_step >= self.max_phase_steps

        toggle = toggle_hit | toggle_timeout

        if torch.any(toggle):
            # 0<->1 토글
            self._phase[toggle] = 1 - self._phase[toggle]
            # 카운터 리셋
            self._near_count[toggle] = 0
            self._phase_step[toggle] = 0
