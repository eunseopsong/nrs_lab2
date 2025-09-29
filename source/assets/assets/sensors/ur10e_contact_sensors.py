# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

@configclass
class UR10eContactSensorsCfg:
    """Contact sensor configuration for UR10e robot."""

    wrist_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )

