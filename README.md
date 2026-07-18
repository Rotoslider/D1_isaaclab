# NavBot D1 — Isaac Lab / robot_lab port (preserved reference)

The D1→Isaac Sim port of **Frank's custom open-source D1** (48:1, high-torque; ~1.03 m/s; Damiao
DM-J6248P-2EC). Added to the **`robot_lab` program** (`fan-ziqi/robot_lab`) as a robot `navbot_d1`,
task `RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0`.

> ⚠️ **Two naming traps** (full explanation in `../docs/05_PORTING_TO_ISAACSIM_FILEMAP.md`):
> 1. This is **not** the built-in **Agibot D1** (`data/Robots/agibot/…`, ~3 m/s) — different robot;
>    never train it.
> 2. Frank's `D1_rl_sar/.../policy/d1/**robot_lab**/` folder is an **Isaac Gym** deploy artifact
>    (ONNX producer = PyTorch 1.10), **not** the `robot_lab` program. There is **no** Frank Isaac Lab
>    config — we own this port.

## Status (2026-07-18): rolled back to a clean slate — this is the preserved reference
The first port trained + moved on Isaac Sim 6.0.1 but ended up **worse than the Isaac Gym baseline**
and never reproduced Frank's gait. It was removed from `robot_lab6` on the box; **these files are the
saved reference + lessons** to restart the port cleanly. `robot_lab6` remains installed and boots
stock tasks (Go2) — it just has no D1 in it now.

## Contents
- **`current_6.0/`** — the port files exactly as they were on Isaac Sim 6.0.1 / Isaac Lab 3.0:
  `navbot_d1.py` (articulation, w/ armature 0.13108), `config/quadruped/navbot_d1/**`,
  `mdp/frank_rewards.py` (Frank's reward formulas in the Lab API), `only_positive_env.py`.
- **`migration_6.0/`** — the platform lessons: `MIGRATION_NOTES.md` (the 9 fixes to make robot_lab
  run on 6.0.1 + the gait fixes), `robot_lab6_port.diff` (full diff), `install_robotlab6.sh`, and
  diagnostic scripts (`d1_diag_pose.py`, `diag_rewards.py`, `d1_render6.py`, `dump_d1_prims.py`).

## Restart the port
See **`../docs/05_PORTING_TO_ISAACSIM_FILEMAP.md`** — §3 (what each file is + where it deploys),
§4 (the clean-restart recipe), §5 (the lessons to apply). In short: copy the URDF/meshes +
`current_6.0/` files into `~/robot_lab6`, re-add `from .frank_rewards import *` to `mdp/__init__.py`,
then train.

## What "faithful" means here (the open problem)
The gait quality — not the plumbing — is what's unsolved. Frank's real gait is **AMP** on **HIM**
(neither ported: AMP deferred, HIM absent from robotlab6's rsl_rl → plain PPO). Actuator PD gains +
armature are **provisional** until the physical 48:1 motors are measured. **Success bar:** the Sim
policy must walk **as well as or better than** the Isaac Gym baseline
(`D1_HIMLoco/.../logs/rough_d1/exported/`).
