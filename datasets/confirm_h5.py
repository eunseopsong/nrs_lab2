import h5py
import os

def inspect_h5(file_path: str):
    """HDF5 파일의 dataset key, shape, dtype을 출력"""
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    print(f"\n📂 파일: {file_path}")
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        print(f"  🔑 Dataset keys: {keys}")
        for key in keys:
            data = f[key]
            print(f"    • {key:20s} shape={data.shape}, dtype={data.dtype}")

if __name__ == "__main__":
    base_dir = "/home/eunseop/nrs_lab2/datasets"
    files = [
        os.path.join(base_dir, "joint_recording_filtered.h5"),
        os.path.join(base_dir, "hand_g_recording.h5"),
    ]

    print("========== HDF5 Dataset Structure ==========")
    for file in files:
        inspect_h5(file)
    print("============================================")
