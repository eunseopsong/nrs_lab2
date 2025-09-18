from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, EventTermCfg as EventTerm
import isaaclab.envs.mdp.observations as base_obs
import nrs_lab2.nrs_lab2.tasks.manager_based.nrs_lab2.mdp.rewards as local_rewards

@configclass
class UR10eSpindleEnvCfg(ManagerBasedRLEnvCfg):
    """UR10e + spindle behavior cloning 환경"""

    # ✅ scene 초기화 (이게 없어서 MISSING_TYPE 에러 났던 것)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)

    # ---------- Observations ----------
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(func=base_obs.joint_pos)
            joint_vel = ObsTerm(func=base_obs.joint_vel)
        policy: PolicyCfg = PolicyCfg()

    observations: ObservationsCfg = ObservationsCfg()

    # ---------- Rewards ----------
    @configclass
    class RewardsCfg:
        dummy = RewTerm(func=local_rewards.bc_zero_reward, weight=0.0)
    rewards: RewardsCfg = RewardsCfg()

    # ---------- Terminations ----------
    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=lambda env, scene: env.common_step_counter >= env.max_episode_length, time_out=True)
    terminations: TerminationsCfg = TerminationsCfg()

    # ---------- Events ----------
    events: EventTerm = EventTerm(func=lambda env, scene: None, mode="reset")

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
