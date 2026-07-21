#!/bin/bash
# Live D1 walk viewer on nuc1 — shows the LATEST AMP-trained policy by default.
# Usage: launch_d1_viewer_nuc1.sh [load_run] [num_envs] [vx]
# Desktop button: ~/Desktop/D1_Walk_Viewer.desktop
# Close the viewer window to exit. Default = fixed 0.6 m/s forward command.
RUN="${1:-}"
ENVS="${2:-16}"
VX="${3:-0.6}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate robotlab6
export OMNI_KIT_ACCEPT_EULA=yes
cd "$HOME/robot_lab6" || exit 1
ARGS=(--task RobotLab-Isaac-Velocity-AMP-NavBot-D1-v0 --num_envs "$ENVS" --vx "$VX" --viz kit)
if [ -n "$RUN" ]; then
  ARGS+=(--load_run "$RUN")
fi
exec python scripts/reinforcement_learning/rsl_rl/d1_view.py "${ARGS[@]}"
