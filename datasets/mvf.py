import numpy as np

def process_joint_recording(input_txt, output_txt, window_size=50, scale_factor=5):
    # 1. txt 불러오기
    data = np.loadtxt(input_txt)

    # 2. 무빙에버리지 필터
    cumsum = np.cumsum(np.insert(data, 0, 0, axis=0), axis=0)
    filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # 앞뒤 window_size 행 제거
    trimmed = filtered[window_size:-window_size]

    # 3. 선형보간으로 길이 5배 확장
    n, d = trimmed.shape
    x_old = np.linspace(0, 1, n)              # 원래 index
    x_new = np.linspace(0, 1, n * scale_factor)  # 확대된 index

    interpolated = np.zeros((len(x_new), d))
    for i in range(d):
        interpolated[:, i] = np.interp(x_new, x_old, trimmed[:, i])

    # 결과 저장
    np.savetxt(output_txt, interpolated, fmt="%.6f")
    print(f"✅ '{output_txt}' 저장 완료! (원래 {n}행 → 보간 후 {interpolated.shape[0]}행)")

if __name__ == "__main__":
    input_file = "joint_recording.txt"
    output_file = "joint_recording_filtered.txt"
    process_joint_recording(input_file, output_file, window_size=50, scale_factor=5)

