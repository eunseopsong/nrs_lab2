# SPDX-License-Identifier: BSD-3-Clause
"""
Convert TXT file (joint data) to HDF5 dataset
for Isaac Lab training.
"""

import numpy as np
import h5py

def txt_to_hdf5(txt_path, h5_path, dataset_key="joint_positions", num_joints=6):
    """
    Convert txt file with joint data to hdf5 format.
    
    Args:
        txt_path (str): Input txt file path.
        h5_path (str): Output h5 file path.
        dataset_key (str): Key name for dataset inside h5.
        num_joints (int): Number of joints (columns in txt).
    """
    # Load txt file: assume whitespace-separated columns
    data = np.loadtxt(txt_path, dtype=np.float32)
    
    # Validate shape
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != num_joints:
        raise ValueError(f"Expected {num_joints} columns, but got {data.shape[1]}")
    
    print(f"Loaded {data.shape[0]} samples with {data.shape[1]} joints.")
    
    # Save to hdf5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(dataset_key, data=data)
    
    print(f"Converted dataset saved to {h5_path} with key '{dataset_key}'.")

if __name__ == "__main__":
    txt_file = "/home/eunseop/nrs_lab2/datasets/hand_g_recording.txt"
    h5_file = "/home/eunseop/nrs_lab2/datasets/hand_g_recording.h5"
    txt_to_hdf5(txt_file, h5_file, dataset_key="joint_positions", num_joints=6)

