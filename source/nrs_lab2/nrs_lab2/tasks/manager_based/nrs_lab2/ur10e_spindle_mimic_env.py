from isaaclab.envs import MimicEnv
from .config.ur10e_spindle.ur10e_spindle_mimic_env_cfg import UR10eSpindleMimicEnvCfg

class UR10eSpindleMimicEnv(MimicEnv):
    cfg_cls = UR10eSpindleMimicEnvCfg

    def __init__(self, cfg: UR10eSpindleMimicEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

