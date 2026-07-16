# NavBot D1 — Isaac Lab / robot_lab port

Ports the **D1 (48:1, high-torque)** quadruped from Isaac Gym / HIMLoco to **Isaac Lab / robot_lab**
so it can train in the newer physics engine. Cloned from robot_lab's **Unitree Go2** template
(identical joint naming: `FL/FR/RL/RR_{hip,thigh,calf}_joint`, base link `base`, foot `.*_foot`)
with our D1's URDF + **48:1 actuator params**.

## Status (2026-07-16)
Config **complete and correct**. Currently blocked on a broken `robotlab` **Isaac Sim 4.5 env**
(the URDF-importer extension doesn't load) — the *stock* Go2 task fails identically, which confirms
the problem is the environment, not this config. Fix = clean reinstall of the Isaac Sim 4.5 /
Isaac Lab 2.3.2 stack, then this trains as-is.

## Files → deploy paths (into robot_lab v2.3.2, `source/robot_lab/robot_lab/`)
| this repo | deploys to |
|---|---|
| `assets/navbot_d1.py` | `assets/navbot_d1.py` |
| `config/quadruped/navbot_d1/**` | `tasks/manager_based/locomotion/velocity/config/quadruped/navbot_d1/**` |

Also: the D1 `urdf/` + `meshes/` → `source/robot_lab/data/Robots/navbot/d1/` (relative mesh paths).
`deploy.sh` copies all of it to the box.

## Task
`RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0`

```bash
cd ~/robot_lab
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 --headless --num_envs 4096
```
Auto-discovered by robot_lab's `import_packages()` — no parent-file edits needed.

## ⚠️ Provisional actuator params — verify after the robot arrives
The 48:1 values in `assets/navbot_d1.py` (kp 50/55/55, kd 3.2/3.5/3.5, effort 50 Nm, vel 6.28 rad/s,
default pose hip ±0.05 / thigh −0.75 / calf −0.75) come from the deployed rl_sar `config.yaml` — a
**starting point**. Re-check against the **measured** motor torque-speed curve once the physical D1
arrives (measure via the Damiao actuator CAN feedback + a Fluke; a DC clamp meter is optional).
Then widen the domain randomization so the trained policy tolerates the real range.

## Notes
- First test-train may need a tweak or two (joint-pose sign, a reward-term name) — normal for a port.
- `flat_env_cfg` not yet added (rough only); trivial to add later (inherits rough, sets flat terrain).
