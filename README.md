# NavBot D1 — Isaac Lab / robot_lab port

Ports **Frank's custom open-source D1** (48:1, high-torque; ~1.03 m/s; Damiao DM-J6248P-2EC) from
Isaac Gym / HIMLoco into the **`robot_lab` program** (`fan-ziqi/robot_lab`, an Isaac Lab extension).
Added as a new robot `navbot_d1`, task `RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0`.

> ⚠️ **Two naming traps** (full explanation in `../docs/05_PORTING_TO_ISAACSIM_FILEMAP.md`):
> 1. This is **not** the built-in **Agibot D1** (`data/Robots/agibot/…`, ~3 m/s) — that's a
>    different robot; never train it.
> 2. Frank's `D1_rl_sar/.../policy/d1/**robot_lab**/` folder is an **Isaac Gym** deploy artifact
>    (ONNX producer = PyTorch 1.10), **not** the `robot_lab` program. There is **no** Frank Isaac
>    Lab config — we own this port.

## Current target: Isaac Sim 6.0.1 / Isaac Lab 3.0 (env `robotlab6`, py3.12)
Earlier attempts: Isaac Sim 4.5.0 (URDF importer won't load — dead end); Isaac Sim 5.1 (worked, but
its RTX renderer segfaults on this dual-GPU box). **6.0.1 is the working target** — RTX viewport
renders, D1 trains. See `migration_6.0/MIGRATION_NOTES.md` for the 9 port fixes.

## Where the files are
- **Authoritative, current copy = on the box:** `~/robot_lab6/source/robot_lab/robot_lab/…`
  (assets/navbot_d1.py, the `navbot_d1/` task dir, `mdp/frank_rewards.py`, `only_positive_env.py`)
  + robot data at `.../data/Robots/navbot/d1/{urdf,meshes}/`.
- **Local backup of the current copy:** `current_6.0/` (synced from the box).
- **Original 5.1-era snapshot:** `assets/`, `config/` (kept for history; superseded by `current_6.0/`).
- **Migration toolkit:** `migration_6.0/` (notes, full `robot_lab6_port.diff`, install + diag scripts).

Full file map (incl. the URDF/reference configs it ports *from*): `../docs/05_PORTING_TO_ISAACSIM_FILEMAP.md`.

## Run it
```bash
conda activate robotlab6 && export OMNI_KIT_ACCEPT_EULA=YES && cd ~/robot_lab6
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0 --headless --num_envs 4096
# watch a run (GUI on the box monitor):  add  --load_run <ts> --viz kit  and first
#   export XAUTHORITY=/run/user/1000/gdm/Xauthority
```

## Status — trains + moves, NOT yet faithful
The D1 spawns, trains (4096 envs), renders, and a best run
(`logs/rsl_rl/navbot_d1_rough/2026-07-16_22-08-32`) walks ~0.78–0.90 m/s on rough terrain. But it is
**not a faithful reproduction of Frank's deployed gait**:
- Frank's real gait is **AMP** (Adversarial Motion Priors) — **not ported** (deferred, large).
- Frank's stack uses **HIM**; robotlab6's rsl_rl has no HIM → we use **plain PPO**.
- Reward formulas are Frank's `D1RoughCfg` re-implemented in `frank_rewards.py`; the only-positive
  clip (`only_positive_env.py`) was the key to stop a frozen policy, but weights still need polish.
- Actuator params (kp 50/55/55, kd 3.2/3.5, effort 50 Nm, **armature 0.13108**) are **provisional** —
  reality-check against **measured** 48:1 motor curves once the physical robot arrives.

Treat the current policy as a working scaffold, not the finished D1 policy.
