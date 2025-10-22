# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for UR10e spindle environment.
- Integrated with nrs_ik_core (C++ FK module)
- Horizon-based trajectory loaders (joints / positions)
- Includes EE pose (x, y, z, roll, pitch, yaw), contact, and camera sensors
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from nrs_ik_core import IKSolver  # ✅ pybind11 기반 C++ 모듈


# ------------------------------------------------------
# Global buffers
# ------------------------------------------------------
_hdf5_joints = None
_hdf5_positions = None
_step_idx = 0



# ------------------------------------------------------
# Quaternion → Euler XYZ 변환 (roll, pitch, yaw)
# ------------------------------------------------------
def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (x, y, z, w) → Euler angles (roll, pitch, yaw)
    - Input: (N, 4)
    - Output: (N, 3)
    """
    x, y, z, w = quat.unbind(-1)

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack((roll, pitch, yaw), dim=-1)


# ------------------------------------------------------
# ✅ EE pose observation (x, y, z, roll, pitch, yaw)
# ------------------------------------------------------
def get_ee_pose(env: "ManagerBasedRLEnv", asset_name: str = "robot", frame_name: str = "wrist_3_link"):
    """Returns end-effector pose (x, y, z, roll, pitch, yaw)."""
    robot = env.scene[asset_name]
    link_indices = robot.find_bodies(frame_name)
    if len(link_indices) == 0:
        raise ValueError(f"[ERROR] Link '{frame_name}' not found in robot asset.")
    idx = link_indices[0]

    ee_pos = robot.data.body_pos_w[:, idx, :]    # (num_envs, 3)
    ee_quat = robot.data.body_quat_w[:, idx, :]  # (num_envs, 4)
    ee_rpy = quat_to_euler_xyz(ee_quat)          # (num_envs, 3)

    # ✅ 강제로 (num_envs, 6) 형태 보장
    ee_pose = torch.cat([ee_pos, ee_rpy], dim=-1)
    if ee_pose.ndim == 3:
        ee_pose = ee_pose.squeeze(1)
    elif ee_pose.ndim == 1:
        ee_pose = ee_pose.unsqueeze(0)
    assert ee_pose.ndim == 2 and ee_pose.shape[1] == 6, f"[EE_POSE] Invalid shape: {ee_pose.shape}"

    return ee_pose




# ------------------------------------------------------
# HDF5 loader: Joints
# ------------------------------------------------------
def load_hdf5_joints(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "target_joints"):
    """HDF5 trajectory (joint targets) 데이터를 로드"""
    global _hdf5_joints, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5 (joints): '{dataset_key}' not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # shape [T, D]

    _hdf5_joints = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 joints of shape {_hdf5_joints.shape} from {file_path}")


# ------------------------------------------------------
# HDF5 loader: Positions
# ------------------------------------------------------
def load_hdf5_positions(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "target_positions"):
    """HDF5 trajectory (position targets) 데이터를 로드"""
    global _hdf5_positions, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5 (positions): '{dataset_key}' not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # shape [T, D]

    _hdf5_positions = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 positions of shape {_hdf5_positions.shape} from {file_path}")


# ------------------------------------------------------
# Observation: target joints (horizon-based)
# ------------------------------------------------------
def get_hdf5_target_joints(env: ManagerBasedRLEnv, horizon: int = 5) -> torch.Tensor:
    """현재 step 기준 horizon 길이만큼 joint target 반환"""
    global _hdf5_joints
    if _hdf5_joints is None:
        D = env.scene["robot"].num_joints
        return torch.zeros((env.num_envs, horizon * D), device=env.device, dtype=torch.float32)

    T, D = _hdf5_joints.shape
    step = env.episode_length_buf[0].item()
    E = env.max_episode_length
    idx = min(int(step / E * T), T - 1)
    future_idx = torch.clamp(torch.arange(idx, idx + horizon), max=T - 1)

    # flatten (horizon, D) → (1, horizon * D)
    future_targets = _hdf5_joints[future_idx].reshape(1, horizon * D)
    return future_targets.repeat(env.num_envs, 1)


# ------------------------------------------------------
# Observation: target positions (horizon-based)
# ------------------------------------------------------
def get_hdf5_target_positions(env: ManagerBasedRLEnv, horizon: int = 5) -> torch.Tensor:
    """현재 step 기준 horizon 길이만큼 position target 반환 (x, y, z, r, p, y 포함)"""
    global _hdf5_positions
    if _hdf5_positions is None:
        D = 6  # position (x,y,z,roll,pitch,yaw)
        return torch.zeros((env.num_envs, horizon * D), device=env.device, dtype=torch.float32)

    T, D = _hdf5_positions.shape
    step = env.episode_length_buf[0].item()
    E = env.max_episode_length
    idx = min(int(step / E * T), T - 1)
    future_idx = torch.clamp(torch.arange(idx, idx + horizon), max=T - 1)

    # flatten (horizon, D) → (1, horizon * D)
    future_targets = _hdf5_positions[future_idx].reshape(1, horizon * D)
    return future_targets.repeat(env.num_envs, 1)


# ------------------------------------------------------
# ✅ Observation: Contact Sensor Forces
# ------------------------------------------------------
def get_contact_forces(env, sensor_name="contact_forces"):
    """Mean contact wrench [Fx, Fy, Fz, 0, 0, 0]"""
    sensor = env.scene.sensors[sensor_name]
    forces_w = sensor.data.net_forces_w
    mean_force = torch.mean(forces_w, dim=1)
    zeros_torque = torch.zeros_like(mean_force)
    contact_wrench = torch.cat([mean_force, zeros_torque], dim=-1)

    step = int(env.common_step_counter)
    if step % 100 == 0:
        fx, fy, fz = mean_force[0].tolist()
        print(f"[ContactSensor DEBUG] Step {step}: Fx={fx:.3f}, Fy={fy:.3f}, Fz={fz:.3f}")

    return contact_wrench


# ------------------------------------------------------
# ✅ Camera distance & normals
# ------------------------------------------------------
def get_camera_distance(env, sensor_name="camera", debug_interval: int = 100):
    """Compute mean camera depth (distance-to-image-plane)."""
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Camera sensor '{sensor_name}' not found in scene.sensors.")
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data.output.get("distance_to_image_plane", None)
    if data is None:
        raise RuntimeError("[ERROR] Missing 'distance_to_image_plane' in camera data output.")
    valid_mask = torch.isfinite(data) & (data > 0)
    valid_data = torch.where(valid_mask, data, torch.nan)
    mean_distance = torch.nanmean(valid_data.view(valid_data.shape[0], -1), dim=1).unsqueeze(1)

    if env.common_step_counter % debug_interval == 0:
        md_cpu = mean_distance[0].detach().cpu().item()
        print(f"[Step {env.common_step_counter}] Mean camera distance: {md_cpu:.4f} m")

    return mean_distance


def get_camera_normals(env, sensor_name="camera"):
    """Compute mean surface normal (x, y, z) from the camera."""
    cam_sensor = env.scene.sensors[sensor_name]
    normals = cam_sensor.data.output.get("normals", None)
    if normals is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    normals_mean = normals.mean(dim=(1, 2))
    if env.common_step_counter % 100 == 0:
        print(f"[Camera DEBUG] Step {env.common_step_counter}: Mean surface normal = {normals_mean[0].cpu().numpy()}")
    return normals_mean
