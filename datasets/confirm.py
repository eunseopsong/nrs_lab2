import h5py
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. HDF5 파일에서 joint trajectory 불러오기
# ================================
with h5py.File("joint_recording.h5", "r") as f:
    q_traj = f["/joint_positions"][:]   # (T, 6) 데이터

print("Trajectory shape:", q_traj.shape)  # (T, 6)

# ================================
# 2. obs–action 쌍 생성
# ================================
obs = q_traj[:-1]      # 현재 상태 (0 ~ T-2)
actions = q_traj[1:]   # 다음 상태 (1 ~ T-1)

print("Obs shape:", obs.shape)        # (T-1, 6)
print("Action shape:", actions.shape) # (T-1, 6)

# ================================
# 3. 새로운 HDF5 파일에 저장
# ================================
with h5py.File("bc_dataset.h5", "w") as f:
    f.create_dataset("observations", data=obs)
    f.create_dataset("actions", data=actions)

print("Saved bc_dataset.h5 with obs/actions datasets.")

# ================================
# 4. 확인 기능 (샘플 몇 개 출력 + 시각화)
# ================================
print("\n[Sample check]")
for i in range(3):
    print(f"t={i}: obs={obs[i]}, action={actions[i]}")

# 시각적으로 obs vs action 비교
plt.figure(figsize=(8, 4))
plt.plot(obs[:100, 0], label="obs q1 (t)")
plt.plot(actions[:100, 0], label="action q1 (t+1)")
plt.title("Obs vs Action Example (Joint q1)")
plt.xlabel("Time step")
plt.ylabel("Joint angle (rad)")
plt.legend()
plt.grid(True)
plt.show()

