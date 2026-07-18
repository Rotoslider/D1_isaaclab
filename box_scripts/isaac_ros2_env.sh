#!/bin/bash
# Point Isaac Sim at its BUNDLED ROS 2 Humble libraries (no system ROS 2 install needed).
# Fixes: "[Error] [isaacsim.ros2.core.impl.extension] ROS2 Bridge startup failed"
#   root cause: AMENT_PREFIX_PATH unset -> the bridge tried a system ROS 2 that isn't installed
#   and couldn't locate rmw. Isaac Sim ships the libs; we just point LD_LIBRARY_PATH at them.
# Box is Ubuntu 22.04 -> use Humble (jazzy is for 24.04). Verified: libs fully resolve here.
# SOURCE this from an Isaac Sim launcher. Do NOT put these in ~/.bashrc -- a persistent
# LD_LIBRARY_PATH to these libs conflicts with real ROS 2 tools (rviz2 etc.) per NVIDIA docs.
_ros2_core="$HOME/miniconda3/envs/robotlab6/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core"
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
case ":$LD_LIBRARY_PATH:" in
  *":$_ros2_core/humble/lib:"*) ;;                        # already present -> don't double-add
  *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_ros2_core/humble/lib" ;;
esac
