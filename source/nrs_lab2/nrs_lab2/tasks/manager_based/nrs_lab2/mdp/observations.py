# SPDX-License-Identifier: BSD-3-Clause
"""
Observation utilities for UR10e spindle environment.
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
def load_hdf5_trajectory(env: ManagerBasedRLEnv, env_ids, file_path: str, dataset_key: str = "joint_positions"):
    """HDF5 trajectory 데이터를 로드 (reset 시 1회 호출)"""
    global _hdf5_trajectory, _step_idx
    import h5py

    with h5py.File(file_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"[ERROR] HDF5: {dataset_key} not found. Available keys: {list(f.keys())}")
        data = f[dataset_key][:]  # [T, D]

    _hdf5_trajectory = torch.tensor(data, dtype=torch.float32, device=env.device)
    _step_idx = 0
    print(f"[INFO] Loaded HDF5 trajectory of shape {_hdf5_trajectory.shape}")


# ------------------------------------------------------
# Observation: next target joints
# ------------------------------------------------------
def get_hdf5_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 step에 해당하는 target joint 반환"""
    global _hdf5_trajectory, _step_idx
    if _hdf5_trajectory is None:
        raise RuntimeError("HDF5 trajectory not loaded. Did you register load_hdf5_trajectory?")

    T = _hdf5_trajectory.shape[0]
    E = env.max_episode_length
    step = env.episode_length_buf[0].item()
    idx = min(int(step / E * T), T - 1)

    return _hdf5_trajectory[idx]

# ------------------------------------------------------
# Observation: future target joints
# ------------------------------------------------------
def get_hdf5_target_future(env: ManagerBasedRLEnv, horizon: int = 5) -> torch.Tensor:
    global _hdf5_trajectory, _step_idx
    if _hdf5_trajectory is None:
        # dummy: [num_envs, horizon * D]
        D = env.scene["robot"].num_joints
        return torch.zeros((env.num_envs, horizon * D), device=env.device, dtype=torch.float32)

    T, D = _hdf5_trajectory.shape
    step = env.episode_length_buf[0].item()
    E = env.max_episode_length
    idx = min(int(step / E * T), T - 1)

    # horizon 만큼 future target 뽑기
    future_idx = torch.clamp(torch.arange(idx, idx + horizon), max=T - 1)

    # shape (horizon, D) → (1, horizon * D) → (num_envs, horizon * D)
    future_targets = _hdf5_trajectory[future_idx].reshape(1, horizon * D)
    return future_targets.repeat(env.num_envs, 1)

# ------------------------------------------------------
# ✅ Observation: Contact Sensor Forces
# ------------------------------------------------------

def get_contact_forces(env, sensor_name="contact_forces"):
    """
    Returns mean contact wrench [Fx, Fy, Fz, 0, 0, 0] for debugging and RL observation.
    """
    sensor = env.scene.sensors[sensor_name]
    data = sensor.data

    # World-frame contact forces: (num_envs, num_bodies, 3)
    forces_w = data.net_forces_w
    mean_force = torch.mean(forces_w, dim=1)  # (num_envs, 3)
    zeros_torque = torch.zeros_like(mean_force)
    contact_wrench = torch.cat([mean_force, zeros_torque], dim=-1)  # (num_envs, 6)

    # ✅ Debug print every few steps (optional)
    step = int(env.common_step_counter)
    if step % 100 == 0:  # every 100 steps
        fx, fy, fz = mean_force[0].tolist()
        print(f"[ContactSensor DEBUG] Step {step}: Fx={fx:.3f}, Fy={fy:.3f}, Fz={fz:.3f}")

    return contact_wrench

# ============================================================
# Camera distance observation (with debug)
# ============================================================

def get_camera_distance(env, sensor_name="camera", debug_interval: int = 100) -> torch.Tensor:
    """
    Retrieve depth (distance-to-image-plane) data from a camera sensor.
    Returns: (num_envs, 1) tensor of mean distances [m]
    """
    # 1️⃣ 카메라 센서 객체 가져오기
    if sensor_name not in env.scene.sensors:
        raise KeyError(f"[ERROR] Camera sensor '{sensor_name}' not found in scene.sensors.")

    sensor = env.scene.sensors[sensor_name]

    # 2️⃣ 카메라 출력 텐서 접근
    data = sensor.data.output.get("distance_to_image_plane", None)
    if data is None:
        raise RuntimeError("[ERROR] Camera data output 'distance_to_image_plane' is missing.")

    # 3️⃣ 유효한 픽셀만 추출
    valid_mask = torch.isfinite(data) & (data > 0)
    valid_data = torch.where(valid_mask, data, torch.nan)

    # 4️⃣ 환경별 평균 거리 계산 (NaN 무시)
    mean_distance = torch.nanmean(valid_data.view(valid_data.shape[0], -1), dim=1)
    mean_distance = mean_distance.unsqueeze(1)  # shape: (num_envs, 1)

    # 5️⃣ 디버깅 (주기적 출력)
    if env.common_step_counter % debug_interval == 0:
        md_cpu = mean_distance[0].detach().cpu().item()
        print(f"[Step {env.common_step_counter}] Mean camera distance: {md_cpu:.4f} m")

        # (선택) 특정 프레임 저장
        if env.common_step_counter % (debug_interval * 10) == 0:
            import matplotlib.pyplot as plt
            import os
            save_dir = os.path.expanduser("~/nrs_lab2/outputs/camera")
            os.makedirs(save_dir, exist_ok=True)

            # depth 이미지 저장
            depth_img = data[0].detach().cpu().numpy()
            plt.imshow(depth_img, cmap="plasma")
            plt.colorbar(label="Distance [m]")
            plt.title(f"Camera Distance (Step {env.common_step_counter})")
            save_path = os.path.join(save_dir, f"camera_distance_step{env.common_step_counter}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[INFO] Saved depth image: {save_path}")

    return mean_distance

def get_camera_normals(env, sensor_name="camera"):
    """
    Retrieve mean surface normal (x, y, z) from the camera sensor.
    """
    cam_sensor = env.scene.sensors[sensor_name]
    normals = cam_sensor.data.output["normals"]
    if normals is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    # (H, W, 3) → (3,)
    normals_mean = normals.mean(dim=(1, 2))  # 전체 이미지 평균 법선
    if env.common_step_counter % 100 == 0:
        print(f"[Camera DEBUG] Step {env.common_step_counter}: Mean surface normal = {normals_mean[0].cpu().numpy()}")

    return normals_mean
