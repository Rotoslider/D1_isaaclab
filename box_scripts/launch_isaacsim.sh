#!/bin/bash
# Full Isaac Sim 6.0 editor GUI
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate robotlab6
export OMNI_KIT_ACCEPT_EULA=yes

# Use Isaac Sim's bundled ROS 2 Humble libs so the ROS 2 Bridge loads (no system ROS 2 needed).
source "$HOME/d1_robot/scripts/isaac_ros2_env.sh"

cd "$HOME"
exec isaacsim isaacsim.exp.full.kit
