import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from lstm import LSTMPolicy  # 같은 폴더에 있는 LSTM 모델 import


# ==============================
# 1️⃣ 데이터 불러오기
# ==============================
def load_dataset(file_path):
    with h5py.File(file_path, "r") as f:
        joint_pos = np.array(f["joint_positions"])
    return joint_pos


# ==============================
# 2️⃣ 예측 함수
# ==============================
def predict_trajectory(model, joint_pos, seq_len=10):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(joint_pos) - seq_len):
            X = torch.tensor(joint_pos[i : i + seq_len], dtype=torch.float32).unsqueeze(0)
            y_pred = model(X).squeeze(0).numpy()
            preds.append(y_pred)
    return np.array(preds)


# ==============================
# 3️⃣ 시각화 함수
# ==============================
def visualize_results(joint_pos, preds, num_joints=6, save_path=None):
    t = np.arange(len(preds))
    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2 * num_joints), sharex=True)

    for j in range(num_joints):
        axes[j].plot(t, joint_pos[10:10+len(preds), j], label="Ground Truth", color='black', linewidth=1.2)
        axes[j].plot(t, preds[:, j], label="LSTM Prediction", color='orange', linestyle='--')
        axes[j].set_ylabel(f"Joint {j+1} [rad]")
        axes[j].grid(True)
        if j == 0:
            axes[j].legend(loc="upper right")
    axes[-1].set_xlabel("Time steps")

    plt.suptitle("LSTM BC Model Prediction vs Ground Truth", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()


# ==============================
# 4️⃣ 실행 메인
# ==============================
def main():
    dataset_path = "/home/eunseop/nrs_lab2/datasets/joint_recording.h5"
    model_path = "/home/eunseop/nrs_lab2/datasets/model_bc.pt"

    # Load dataset
    joint_pos = load_dataset(dataset_path)

    # Load model
    input_dim = joint_pos.shape[1]
    model = LSTMPolicy(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Predict
    preds = predict_trajectory(model, joint_pos)

    # Visualize
    visualize_results(joint_pos, preds, num_joints=input_dim, save_path="/home/eunseop/nrs_lab2/datasets/lstm_prediction.png")


if __name__ == "__main__":
    main()
