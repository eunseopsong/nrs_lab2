# sim_env.py
"""
이 프로젝트는 Isaac Sim + ROS로 데이터를 이미 모아서 hdf5로 쓰고 있으므로
ACT 원본에서 쓰던 dm_control 기반 시뮬 환경은 사용하지 않는다.

다만 일부 스크립트가 `from sim_env import BOX_POSE, make_sim_env`
를 import 하므로, 여기서 같은 이름만 정의해준다.
"""

from custom_constants import DT, START_ARM_POSE

# 원본 코드에서 외부에서 바꾸는 전역이라서 일단 남겨둔다
BOX_POSE = [None]


def make_sim_env(task_name: str):
    """
    Isaac Sim을 쓰는 현재 파이프라인에서는 이 함수를 부를 일이 없다.
    만약 `sim_...` 태스크를 진짜로 돌리고 싶으면,
    여기서 Isaac Sim용 gym/env 래퍼를 만들어서 반환하도록 바꿔라.
    """
    raise NotImplementedError(
        "This project uses Isaac Sim + ROS logs, not dm_control. "
        "Define an Isaac-Sim-backed env here if you really need sim."
    )
