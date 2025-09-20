from isaaclab.envs import MimicEnvCfg
from isaaclab.managers import SceneEntityCfg

class UR10eSpindleMimicEnvCfg(MimicEnvCfg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 로봇 articulation prim (기존 polishing_env_cfg 참고)
        self.scene.robot = SceneEntityCfg(
            prim_path="/World/ur10e_w_spindle_robot",
            spawn=True
        )

        # EE 프레임 (spindle tool frame)
        self.commands.ee_pose.body_name = "wrist_3_link"

        # Demonstration dataset 위치
        self.mimic.data_file = "/home/eunseop/nrs_lab2/datasets/joint_recording.h5"

        # Subtask 정의 (없으면 전체 trajectory 하나로 간주)
        self.mimic.subtask_terms = []

