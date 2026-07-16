#!/bin/bash
# Deploy the NavBot D1 Isaac Lab port into robot_lab on p3tiny.
# Run from the isaaclab_port/ directory: bash deploy.sh
set -e
HOST=p3tiny@192.168.1.197
RLB=/home/p3tiny/robot_lab/source/robot_lab/robot_lab
DATA=/home/p3tiny/robot_lab/source/robot_lab/data/Robots/navbot/d1
QCFG=$RLB/tasks/manager_based/locomotion/velocity/config/quadruped/navbot_d1

echo "== articulation asset =="
scp -q assets/navbot_d1.py "$HOST:$RLB/assets/navbot_d1.py"

echo "== task config =="
ssh "$HOST" "mkdir -p $QCFG/agents"
scp -q config/quadruped/navbot_d1/__init__.py config/quadruped/navbot_d1/rough_env_cfg.py "$HOST:$QCFG/"
scp -q config/quadruped/navbot_d1/agents/__init__.py config/quadruped/navbot_d1/agents/rsl_rl_ppo_cfg.py "$HOST:$QCFG/agents/"

echo "== robot assets (urdf + meshes) — copied from D1_HIMLoco if not present =="
ssh "$HOST" "mkdir -p $DATA && [ -d $DATA/urdf ] || cp -r ~/d1_robot/D1_HIMLoco/legged_gym/resources/robots/d1/urdf ~/d1_robot/D1_HIMLoco/legged_gym/resources/robots/d1/meshes $DATA/"

echo "== done. task: RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 =="
