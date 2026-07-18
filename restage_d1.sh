#!/bin/bash
# restage_d1.sh — re-add the NavBot D1 to robot_lab6 for a fresh Isaac Sim port.
#
# WHAT IT DOES (idempotent — safe to re-run):
#   Copies the preserved D1 port reference + the canonical URDF/meshes into the working
#   robot_lab6 tree on this box, and re-wires the custom rewards import. After it runs, the
#   task `RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0` is registered and trainable.
#
# WHERE IT PULLS FROM (this box):
#   ~/d1_robot/port_ref/                                  <- the port config reference
#     (mirror of the git master: isaaclab_port/current_6.0/ on the nuc1 project folder)
#   ~/d1_robot/D1_HIMLoco/legged_gym/resources/robots/d1/{urdf,meshes}   <- canonical URDF (md5 38566e08)
#
# WHERE IT PUTS FILES (into ~/robot_lab6/source/robot_lab/robot_lab/):
#   assets/navbot_d1.py
#   tasks/manager_based/locomotion/velocity/config/quadruped/navbot_d1/**
#   tasks/manager_based/locomotion/velocity/mdp/frank_rewards.py
#   tasks/manager_based/locomotion/velocity/only_positive_env.py
#   data/Robots/navbot/d1/{urdf,meshes}/
#   ...and adds `from .frank_rewards import *` to .../velocity/mdp/__init__.py if missing.
#
# It does NOT train and does NOT tune — it only stages files. Then:
#   conda activate robotlab6 && export OMNI_KIT_ACCEPT_EULA=YES && cd ~/robot_lab6
#   python scripts/reinforcement_learning/rsl_rl/train.py \
#     --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 --headless --num_envs 4096
set -euo pipefail

REF="$HOME/d1_robot/port_ref"
URDF_SRC="$HOME/d1_robot/D1_HIMLoco/legged_gym/resources/robots/d1"
RL="$HOME/robot_lab6/source/robot_lab/robot_lab"
V="$RL/tasks/manager_based/locomotion/velocity"

# --- sanity checks ---
[ -d "$REF" ]      || { echo "ERROR: port reference not found: $REF"; echo "  (copy isaaclab_port/current_6.0/ here from the nuc1 project)"; exit 1; }
[ -d "$URDF_SRC" ] || { echo "ERROR: URDF source not found: $URDF_SRC"; exit 1; }
[ -d "$RL" ]       || { echo "ERROR: robot_lab6 not found: $RL"; exit 1; }

echo "== staging NavBot D1 into robot_lab6 =="

# 1. robot data (URDF + meshes)
mkdir -p "$RL/data/Robots/navbot/d1"
cp -r "$URDF_SRC/urdf"   "$RL/data/Robots/navbot/d1/"
cp -r "$URDF_SRC/meshes" "$RL/data/Robots/navbot/d1/"
echo "  [ok] URDF + meshes -> data/Robots/navbot/d1/"

# 2. articulation cfg
cp "$REF/navbot_d1.py" "$RL/assets/navbot_d1.py"
echo "  [ok] assets/navbot_d1.py"

# 3. task config dir
mkdir -p "$V/config/quadruped"
rm -rf "$V/config/quadruped/navbot_d1"
cp -r "$REF/config/quadruped/navbot_d1" "$V/config/quadruped/navbot_d1"
echo "  [ok] config/quadruped/navbot_d1/**"

# 4. custom rewards + only-positive env wrapper
cp "$REF/mdp/frank_rewards.py" "$V/mdp/frank_rewards.py"
cp "$REF/only_positive_env.py" "$V/only_positive_env.py"
echo "  [ok] mdp/frank_rewards.py + only_positive_env.py"

# 5. wire the custom reward formulas into the mdp package (idempotent)
IMPORT_LINE='from .frank_rewards import *  # noqa: F401, F403'
if ! grep -qF "from .frank_rewards import" "$V/mdp/__init__.py"; then
  printf '%s\n' "$IMPORT_LINE" >> "$V/mdp/__init__.py"
  echo "  [ok] added frank_rewards import to mdp/__init__.py"
else
  echo "  [skip] frank_rewards import already present in mdp/__init__.py"
fi

# 6. sanity: task must not be blacklisted
if grep -q '"navbot' "$RL/tasks/__init__.py"; then
  echo "  [WARN] 'navbot' appears in the blacklist in tasks/__init__.py — remove it or the task won't register."
fi

echo ""
echo "== staged. Train with: =="
echo "  conda activate robotlab6 && export OMNI_KIT_ACCEPT_EULA=YES && cd ~/robot_lab6"
echo "  python scripts/reinforcement_learning/rsl_rl/train.py \\"
echo "    --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 --headless --num_envs 4096"
echo ""
echo "NOTE: this stages the PRIOR config (which trained worse than the Isaac Gym baseline)."
echo "      Apply the lessons in docs/05_PORTING_TO_ISAACSIM_FILEMAP.md §5 as you tune."
