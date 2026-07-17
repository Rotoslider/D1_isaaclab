# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


# The blacklist prevents importing sub-packages. For the Isaac Lab 3.0 / Isaac Sim 6.0
# port we only need the NavBot D1 quadruped velocity task, so every other task family
# and robot config is blacklisted (substring match on the dotted module path) to avoid
# their 2.3->3.0 API breaks and exotic deps (e.g. cusrl) blocking our import.
_BLACKLIST_PKGS = [
    "utils",
    "beyondmimic",
    "direct",
    "humanoid",
    "wheeled",
    "velocity.config.others",
    "agibot_d1",
    "anymal_d",
    "deeprobotics_lite3",
    "magiclab_magicdog",
    "unitree_a1",
    "unitree_b2",
    "unitree_go2",
    "zsibot_zsl1",
]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
