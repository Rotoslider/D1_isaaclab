#!/bin/bash
# View the Isaac Gym AMP baseline policy (rough_d1_amp) walking on the box monitor.
# Usage: launch_d1_gym_amp_viewer.sh [load_run] [checkpoint]   (default: latest run)
# Viewer keys: F = toggle follow-cam, N = next robot.
RUN="${1:-}"
CKPT="${2:-10000}"
export DISPLAY=:0
export XAUTHORITY=/run/user/1000/gdm/Xauthority
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd ~/d1_robot/D1_HIMLoco/legged_gym/legged_gym/scripts
ARGS=(--task=d1_amp_canonical_cmdcond --num_envs 16)
if [ -n "$RUN" ]; then ARGS+=(--load_run "$RUN" --checkpoint "$CKPT"); fi
exec python play.py "${ARGS[@]}"
