#!/bin/bash
# Open the live Isaac Sim viewer on the box monitor showing a trained D1 run.
# Usage: launch_d1_viewer.sh [run_name] [num_envs]
#   run_name  defaults to the newest run in robot_lab6 logs (e.g. 2026-07-18_22-31-15_frankhist01)
#   num_envs  defaults to 16
# Interactive: mouse-orbit/zoom, runs until you close the window.
LOGDIR=$HOME/robot_lab6/logs/rsl_rl/navbot_d1_rough
RUN=${1:-$(ls -t $LOGDIR | head -1)}
NENVS=${2:-16}
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:0
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate robotlab6
export OMNI_KIT_ACCEPT_EULA=YES
cd $HOME/robot_lab6
echo "Viewing run: $RUN ($NENVS robots) — close the Isaac Sim window to exit."
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 \
  --num_envs $NENVS --load_run "$RUN" --viz kit
