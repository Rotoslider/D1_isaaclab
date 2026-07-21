# NavBot D1 — Isaac Sim 6 / Isaac Lab 3.0 port (training stack + AMP)

The complete Isaac Lab port of **Frank's open-source NavBot D1** quadruped (48:1 gearing,
Damiao DM-J6248P-2EC actuators, ~1.03 m/s top speed) — from Frank's legacy Isaac Gym /
legged_gym stack (`NavBotHub/D1_HIMLoco`) onto **Isaac Sim 6.0.1 + Isaac Lab 3.0 +
`robot_lab`**, including a faithful port of his canonical **AMP (Adversarial Motion
Priors)** training recipe.

## Status (2026-07-21): port complete, at parity with the Gym baseline

Same-protocol head-to-head (all-16-env flat eval), our Isaac Lab port vs the Gym
baseline trained from Frank's own stack on identical recipe + dataset:

| | Gym baseline (38.7 h) | **Isaac Lab port (19.5 h)** |
|---|---|---|
| speed @ cmd 0.5 m/s | 0.48 m/s | **0.47 m/s** |
| speed @ cmd 0.8 m/s | 0.74 m/s | **0.71 m/s** |
| gait cadence vs reference clips | — | **2.24 Hz vs 2.25 Hz** |

Both stacks show the same recipe-inherent ~10% undershoot; the port has tighter
per-env spreads and trains 2× faster wall-clock. The original success bar — the Sim
policy must walk as well as or better than the Isaac Gym baseline — is met. Current
work: rough-terrain AMP stage (warm-started from the flat policy), `rl_sar`
deployment package. Actuator PD gains + armature remain provisional until the
physical 48:1 motors are measured (HIM estimator not ported — plain PPO with
6-frame observation history; documented gap).

> ⚠️ **Two naming traps:**
> 1. This is **not** the built-in **Agibot D1** in upstream robot_lab
>    (`data/Robots/agibot/…`, ~3 m/s) — a different robot entirely; never train it.
> 2. Frank's `D1_rl_sar/.../policy/d1/robot_lab/` folder is an **Isaac Gym** deploy
>    artifact (ONNX producer = PyTorch 1.10), **not** the `robot_lab` program. There
>    is no upstream Isaac Lab config for this robot — this repo owns the port.

## Contents

- `current_6.0/` — the port itself, mirroring robot_lab file layout: `navbot_d1.py`
  asset (48:1 actuators, armature 0.13108), manager-based task configs (rough PPO,
  AMP flat, AMP rough), `mdp/` custom rewards (Frank's D1RoughCfg formulas in
  `frank_rewards.py` + the cmdcond alignment in `cmdcond_rewards.py`),
  `only_positive_env.py`, and `amp/` — Frank's AMP core (discriminator / motion
  loader / replay buffer / normalizer, vendored verbatim: pure torch, no Isaac Gym
  deps) plus our Isaac Lab integration (`AmpVelocityEnv`, `SlewedVelocityCommand`,
  `AmpPPO`, `AmpOnPolicyRunner`, smoke/eval tools).
- `migration_6.0/` — the platform lessons: `MIGRATION_NOTES.md` (the 9 fixes to make
  robot_lab run on 6.0.1), `robot_lab6_port.diff`, install scripts, diagnostics
  (`d1_diag_pose.py`, `diag_rewards.py`, `d1_render6.py`).
- `AMP_PORT_DESIGN.md` — AMP port architecture, decisions, must-replicate list,
  final head-to-head results.
- `CMDCOND_REWARD_MAP.md` — Frank's canonical cmdcond reward table (44 terms)
  mapped against the port, with formula provenance.
- `deploy/` — exported policies + `rl_sar` deploy configs (empirically probed
  observation layout: term-major history, oldest-first, FR/FL/RR/RL joint order,
  unscaled commands).
- `box_scripts/`, `install_robotlab6_nuc1.sh`, `restage_d1.sh` — environment
  install/restage recipes and viewer/eval tooling.

## Reproduce

1. Install the stack: `install_robotlab6_nuc1.sh` (Isaac Sim 6.0.1 pip, Isaac Lab
   3.0, robot_lab, py3.12 + torch 2.11 cu128 — read the post-install fix comments).
2. Stage the D1 into robot_lab: `restage_d1.sh` (URDF + meshes + configs).
3. Train PPO: `train.py --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 --headless`
   • AMP: `--task RobotLab-Isaac-Velocity-AMP-NavBot-D1-v0` (needs the reference
   motion dataset from `NavBotHub/D1_HIMLoco` `datasets/d1_amp_v4_h_uniform_cmdcond`
   copied to `robot_lab/data/Motions/d1/`).

Upstream credits: [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab),
[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab),
[NavBotHub](https://github.com/NavBotHub) (Frank's D1 robot, HIMLoco training stack,
AMP recipe + reference-motion dataset).
