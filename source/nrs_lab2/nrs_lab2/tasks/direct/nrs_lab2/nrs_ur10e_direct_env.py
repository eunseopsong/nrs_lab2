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
    """Direct RL env for UR10e + spindle.
    - Actions: Δq (rad), scaled by cfg.action_scale, applied as position targets
    - Observations: [q(6), dq(6)]
    - Rewards: home-pose tracking (L2) + small velocity penalty
    - Dones: timeout only
    """
    cfg: NrsUR10eDirectEnvCfg

    def __init__(self, cfg: NrsUR10eDirectEnvCfg, render_mode: str | None = None, **kwargs):
        # NOTE: super().__init__() calls _setup_scene() and creates self.robot
        super().__init__(cfg, render_mode, **kwargs)

        # --- Resolve joint indices robustly across API variants ---
        self._jname = list(self.cfg.joint_names)

        jid_raw, _ = self.robot.find_joints(self._jname)
        # jid_raw can be: list[int], tuple[int], torch.Tensor, list[list[int]]
        if isinstance(jid_raw, (list, tuple)):
            # flatten if nested
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
            # single int fallback
            idx_list = [int(jid_raw)]

        assert len(idx_list) == len(self._jname), \
            f"[UR10e] Joint index count mismatch: names={self._jname} -> indices={idx_list}"

        self._jid = torch.as_tensor(idx_list, device=self.device, dtype=torch.long)  # (6,)

        # State refs
        self.q = self.robot.data.joint_pos   # (num_envs, dof)
        self.dq = self.robot.data.joint_vel  # (num_envs, dof)

        # Home pose tensor from articulation cfg
        q_ref_list = [self.robot.cfg.init_state.joint_pos[name] for name in self._jname]
        self.q_ref = torch.tensor(q_ref_list, device=self.device, dtype=self.q.dtype).unsqueeze(0)  # (1,6)

        # Safe default action (before first _pre_physics_step)
        self.actions = torch.zeros((self.num_envs, len(self._jname)), device=self.device, dtype=self.q.dtype)

    # ---------------- Scene ----------------
    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)
        # Ground
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        # Clone envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register
        self.scene.articulations["robot"] = self.robot

        # Lighting
        dl = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        dl.func("/World/Light", dl)

    # ---------------- Actions ----------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Expect (num_envs, 6)
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        dq_cmd = self.actions * self.cfg.action_scale                 # (num_envs, 6)
        q_now = self.q[:, self._jid]                                   # (num_envs, 6)
        q_tgt = q_now + dq_cmd
        self.robot.set_joint_position_target(q_tgt, joint_ids=self._jid)

    # ---------------- Observations ----------------
    def _get_observations(self) -> dict:
        obs = torch.cat((self.q[:, self._jid], self.dq[:, self._jid]), dim=-1)  # (num_envs, 12)
        return {"policy": obs}

    # ---------------- Rewards ----------------
    def _get_rewards(self) -> torch.Tensor:
        err_q = self.q[:, self._jid] - self.q_ref
        rew = -1.0 * torch.sum(err_q * err_q, dim=-1) \
              -0.001 * torch.sum(self.dq[:, self._jid] * self.dq[:, self._jid], dim=-1)
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
