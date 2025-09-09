# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from . import nrs_ur10e_direct_env, nrs_ur10e_direct_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Template-Nrs-Lab2-Direct-v0",
    entry_point=f"{__name__}.nrs_lab2_env:NrsLab2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nrs_lab2_env_cfg:NrsLab2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Nrs-UR10e-Direct-v1",
    entry_point=f"{nrs_ur10e_direct_env.__name__}:NrsUR10eDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{nrs_ur10e_direct_env_cfg.__name__}:NrsUR10eDirectEnvCfg",
        # ↓ skrl 트레이너/플레이어가 읽을 에이전트 설정 경로 추가
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # (선택) RSL-RL도 쓸 거면 같이 연결
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)