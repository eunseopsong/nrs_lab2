#!/usr/bin/env python3
import h5py
import matplotlib.pyplot as plt

def main():
    file_path = "joint_recording.h5"  # 같은 폴더에 있는 경우

    # HDF5 파일 열기
    with h5py.File(file_path, "r") as f:
        if "joint_positions" not in f:
            print("❌ 'joint_positions' dataset not found in HDF5 file.")
            print("Available keys:", list(f.keys()))
            return

        data = f["joint_positions"][:]   # shape: (N,6)

    print(f"Loaded joint_positions with shape {data.shape}")

    # --- 1. Joint trajectories ---
    plt.figure(figsize=(10,6))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=f"q{i+1}")
    plt.xlabel("Time step")
    plt.ylabel("Joint angle (rad)")
    plt.title("Joint Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 2. 3D Visualization (q1,q2,q3 예시) ---
    if data.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(data[:,0], data[:,1], data[:,2], label="q1-q2-q3 trajectory")
        ax.set_xlabel("q1 (rad)")
        ax.set_ylabel("q2 (rad)")
        ax.set_zlabel("q3 (rad)")
        ax.set_title("3D Joint Space Trajectory (q1,q2,q3)")
        ax.legend()
        plt.show()

if __name__ == "__main__":
    main()

