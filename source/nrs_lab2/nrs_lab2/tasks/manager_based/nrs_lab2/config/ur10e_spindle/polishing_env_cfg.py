from isaaclab.utils import configclass
from nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.ur10e_spindle_env_cfg import UR10eSpindleEnvCfg


@configclass
class PolishingPoseHoldEnvCfg(UR10eSpindleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # BC용 trajectory 파일 경로
        self.dataset_path = "/home/eunseop/nrs_lab2/src/rtde_handarm2/data/trajectory.h5"
