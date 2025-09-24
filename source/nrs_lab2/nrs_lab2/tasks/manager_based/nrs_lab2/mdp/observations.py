# SPDX-License-Identifier: BSD-3-Clause
"""
Observation functions for UR10e spindle environment.
- Load and provide HDF5 trajectory targets
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv

# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_hdf5_trajectory = None
_step_idx = 0


# ------------------------------------------------------
# HDF5 trajectory loader
# ------------------------------------------------------
def load_hdf5_trajectory(env: ManagerBasedRLEnv, trajectory: torch.Tensor):
    """Register HDF5 trajectory into global buffer (reset at episode start)."""
    global _hdf5_trajectory, _step_idx
    _hdf5_trajectory = trajectory.clone().to(env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 trajectory of shape {_hdf5_trajectory.shape}")


# ------------------------------------------------------
# Observation: current target joints from trajectory
# ------------------------------------------------------
def get_hdf5_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Return next target joint positions for each env."""
    global _hdf5_trajectory, _step_idx
    if _hdf5_trajectory is None:
        print("[DEBUG] get_hdf5_target: trajectory is None → returning zeros")
        return torch.zeros((len(env_ids), 6), device=env.device)

    targets = []
    for _ in env_ids.tolist():
        if _step_idx < _hdf5_trajectory.shape[0]:
            targets.append(_hdf5_trajectory[_step_idx])
        else:
            targets.append(torch.zeros(6, device=env.device))

    targets = torch.stack(targets, dim=0)
    _step_idx += 1

    return targets
