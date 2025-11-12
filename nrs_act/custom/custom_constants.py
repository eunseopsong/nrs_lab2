### Task parameters
# Author: Chemin Ahn (chemx3937@gmail.com)
# Use of this source code is governed by the MIT, see LICENSE

import os
import math

# C++ 레코더가 저장하는 최상위 디렉토리
#   /home/eunseop/nrs_lab2/datasets/ACT/<MMDD_HHMM>/act_data.hdf5
DATA_DIR = "/home/eunseop/nrs_lab2/datasets/ACT"


def _find_latest_run_dir(base_dir: str) -> str:
    """ACT/ 아래에서 가장 최근에 만든 폴더 하나 골라서 반환."""
    base_dir = os.path.expanduser(base_dir)
    if not os.path.isdir(base_dir):
        return base_dir

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        return base_dir

    # 수정 시간 기준으로 최신 폴더
    return max(subdirs, key=os.path.getmtime)


# 가장 최근 기록 폴더
_latest = _find_latest_run_dir(DATA_DIR)
# 변환 스크립트가 여기 안에 episodes/ 를 만든다고 가정
_default_episode_dir = os.path.join(_latest, "episodes")

TASK_CONFIGS = {
    # 학습/추론 때 --task_name ur10e_swing 으로 부를 이름
    "ur10e_swing": {
        "dataset_dir": _default_episode_dir,
        "num_episodes": 50,
        # demo_data_act_form.py 에서 만든 T_pad에 맞춰서 조절
        "episode_len": 300,
        "camera_names": ["cam_front", "cam_head"],
    },
}

### ALOHA / control 고정 상수
DT = 0.05  # 20Hz

# C++ 레코더에서 실제로 저장하던 UR10e의 이름과 맞춘다
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# 초기 포즈: 0, -90, -90, -90, 90, 0 (deg)
base     = 0.0   * math.pi / 180.0
shoulder = -90.0 * math.pi / 180.0
elbow    = -90.0 * math.pi / 180.0
wrist1   = -90.0 * math.pi / 180.0
wrist2   = 90.0  * math.pi / 180.0
wrist3   = 0.0   * math.pi / 180.0

START_ARM_POSE = [base, shoulder, elbow, wrist1, wrist2, wrist3]
