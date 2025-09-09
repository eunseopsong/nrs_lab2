# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from collections.abc import Sequence
from typing import Tuple
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .nrs_ur10e_direct_env_cfg import NrsUR10eDirectEnvCfg


class NrsUR10eDirectEnv(DirectRLEnv):
    """UR10e + Spindle(TCP 오프셋) 환경: 두 목표점을 번갈아 방문하며, TCP를 수평(-Z)로 유지.

    - Actions: Δq (rad), [-1,1] → action_scale 배 → position target 적층
    - Observations: [q(6), dq(6)]
    - Rewards:
        * 위치 L2 오차 (TCP vs Target A/B)
        * 근접 보너스 (bonus_reach)
        * 페이즈 전환 보너스 (bonus_phase)
        * 페이즈 정체 패널티 (phase_step_penalty * phase_step)
        * 자세 보상: TCP z축이 월드 -Z에 정렬되도록 (w_ori)
        * 속도/액션 패널티 (w_dq, w_act), (옵션) 홈포즈 유지 (w_home)
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
        ee_candidates = []
        if getattr(self.cfg, "ee_frame_name", None):
            ee_candidates.append(self.cfg.ee_frame_name)
        ee_candidates += ["ee_link", "tool0", "flange", "wrist_3_link"]
        ee_found = False
        for name in ee_candidates:
            try:
                bid, _ = self.robot.find_bodies([name])
                if isinstance(bid, (list, tuple)) and len(bid) > 0:
                    self._ee_id = int(bid[0] if not isinstance(bid[0], (list, tuple)) else bid[0][0])
                    self._ee_name = name
                    ee_found = True
                    break
                elif torch.is_tensor(bid):
                    self._ee_id = int(bid.flatten()[0].item())
                    self._ee_name = name
                    ee_found = True
                    break
            except Exception:
                pass
        if not ee_found:
            raise RuntimeError("EE 링크를 찾을 수 없습니다. cfg.ee_frame_name을 확인하세요.")

        # --- State refs ---
        self.q  = self.robot.data.joint_pos   # (num_envs, dof)
        self.dq = self.robot.data.joint_vel   # (num_envs, dof)

        # Home pose (옵션)
        q_ref_list = [self.robot.cfg.init_state.joint_pos[name] for name in self._jname]
        self.q_ref = torch.tensor(q_ref_list, device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,6)

        # --- Targets (TCP 기준, world frame) ---
        self.target_A = torch.tensor([0.5, -0.4, 0.1], device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,3)
        self.target_B = torch.tensor([0.5,  0.4, 0.1], device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,3)

        # --- TCP offset (in EE local frame) ---
        if self.cfg.tcp_offset is None:
            self.tcp_offset = torch.tensor([0.0, 0.0, 0.185], device=self.device, dtype=self.q.dtype)
        else:
            self.tcp_offset = torch.tensor(self.cfg.tcp_offset, device=self.device, dtype=self.q.dtype)

        # --- Orientation target (world) ---
        self._z_local = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.q.dtype)     # EE 로컬 z축
        self._z_world_target = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=self.q.dtype)  # 월드 -Z

        # --- Phase buffers ---
        self._phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 0->A, 1->B
        self._near_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._phase_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # --- Hyper-parameters (cfg에서 넘어온 값 사용) ---
        self.reach_tol = self.cfg.reach_tol
        self.hold_steps = self.cfg.hold_steps
        self.max_phase_steps = self.cfg.max_phase_steps
        self.w_pos = self.cfg.w_pos
        self.w_ori = self.cfg.w_ori
        self.w_dq  = self.cfg.w_dq
        self.w_act = self.cfg.w_act
        self.bonus_reach = self.cfg.bonus_reach
        self.bonus_phase = self.cfg.bonus_phase
        self.phase_step_penalty = self.cfg.phase_step_penalty
        self.enable_home_keep = self.cfg.enable_home_keep
        self.w_home = self.cfg.w_home

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
        # 1) TCP(world) 포즈
        tcp_pos, tcp_quat = self._get_tcp_pose()  # (N,3), (N,4)

        # 2) 페이즈별 타깃 선택
        tgt = torch.where(self._phase.unsqueeze(-1) == 0, self.target_A, self.target_B)  # (N,3)

        # 3) 위치 오차 / 근접 판정
        pos_err = tcp_pos - tgt
        pos_err_sq = torch.sum(pos_err * pos_err, dim=-1)  # (N,)
        near = pos_err_sq <= (self.reach_tol * self.reach_tol)

        # 4) 자세 오차: TCP z축(로컬 [0,0,1])을 월드 -Z로 정렬
        z_axis = self._quat_rotate_xyzw(tcp_quat, self._z_local)       # (N,3)
        cos_val = torch.sum(z_axis * self._z_world_target, dim=-1).clamp(-1.0, 1.0)
        ori_err = 1.0 - cos_val  # 0이면 완벽 정렬

        # 5) 기본 보상
        rew = -self.w_pos * pos_err_sq \
              - self.w_ori * ori_err \
              - self.w_dq  * torch.sum(self.dq[:, self._jid] * self.dq[:, self._jid], dim=-1) \
              - self.w_act * torch.sum(self.actions * self.actions, dim=-1)

        if self.enable_home_keep:
            err_q = self.q[:, self._jid] - self.q_ref
            rew = rew - self.w_home * torch.sum(err_q * err_q, dim=-1)

        # 6) 도달 보너스
        rew = rew + self.bonus_reach * near.to(rew.dtype)

        # 7) 페이즈 업데이트 + 전환 보너스/정체 패널티
        toggle = self._update_phase(near)
        if torch.any(toggle):
            rew = rew + self.bonus_phase * toggle.to(rew.dtype)  # 전환된 env만
        rew = rew - self.phase_step_penalty * self._phase_step.to(rew.dtype)

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
    def _get_tcp_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """TCP world position + quaternion(xyzw) 반환."""
        d = self.robot.data

        # EE world position
        if hasattr(d, "body_pos_w"):
            ee_pos = d.body_pos_w[:, self._ee_id, :]  # (N,3)
        elif hasattr(d, "link_pos_w"):
            ee_pos = d.link_pos_w[:, self._ee_id, :]
        elif hasattr(d, "body_state_w"):
            ee_pos = d.body_state_w[:, self._ee_id, 0:3]
        elif hasattr(d, "link_state_w"):
            ee_pos = d.link_state_w[:, self._ee_id, 0:3]
        else:
            raise RuntimeError("EE 위치 데이터를 찾을 수 없습니다.")

        # EE world orientation (xyzw)
        if hasattr(d, "body_quat_w"):
            ee_quat = d.body_quat_w[:, self._ee_id, :]  # (N,4) xyzw
        elif hasattr(d, "link_quat_w"):
            ee_quat = d.link_quat_w[:, self._ee_id, :]
        elif hasattr(d, "body_state_w"):
            ee_quat = d.body_state_w[:, self._ee_id, 3:7]  # pos(3)+quat(4)
        elif hasattr(d, "link_state_w"):
            ee_quat = d.link_state_w[:, self._ee_id, 3:7]
        else:
            raise RuntimeError("EE 쿼터니언 데이터를 찾을 수 없습니다.")

        # TCP(world) 위치: EE pos + (EE quat로 회전한 로컬 offset)
        offset_world = self._quat_rotate_xyzw(ee_quat, self.tcp_offset)  # (N,3)
        tcp_pos = ee_pos + offset_world
        return tcp_pos, ee_quat

    @staticmethod
    def _quat_rotate_xyzw(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """쿼터니언 q(xyzw)로 벡터 v(3,) 회전. 배치 지원.
        v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + w*v)
        """
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(q_xyzw.size(0), -1)
        q_xyz = q_xyzw[:, 0:3]            # (N,3)
        q_w   = q_xyzw[:, 3:4]            # (N,1)
        t = 2.0 * torch.cross(q_xyz, v, dim=1)
        v_rot = v + q_w * t + torch.cross(q_xyz, t, dim=1)
        return v_rot

    def _update_phase(self, near: torch.Tensor) -> torch.Tensor:
        """목표 근접 연속성/최대 지속시간을 기준으로 A<->B 페이즈 토글.
        Returns:
            toggle (BoolTensor): 이번 스텝에 페이즈가 토글된 env 마스크
        """
        # 연속 근접 카운트
        self._near_count = torch.where(near, self._near_count + 1, torch.zeros_like(self._near_count))
        # 페이즈 경과 스텝
        self._phase_step = self._phase_step + 1

        # 토글 조건
        toggle_hit = self._near_count >= self.hold_steps
        toggle_timeout = self._phase_step >= self.max_phase_steps
        toggle = toggle_hit | toggle_timeout

        if torch.any(toggle):
            self._phase[toggle] = 1 - self._phase[toggle]
            self._near_count[toggle] = 0
            self._phase_step[toggle] = 0

        return toggle
