# NavBot D1 — migration to Isaac Sim 6.0.1 / Isaac Lab 3.0

**Why:** On the p3tiny box (RTX 4000 Ada + Intel iGPU, driver **595.71.05-open**), Isaac Sim
**5.1**'s RTX renderer segfaulted on init (both GUI viewport and offscreen `--video`), so the D1
could only be *trained*, never *seen*, in Isaac Sim. Isaac Sim **6.0.1** lists driver **595.58.03**
as its requirement — the box's 595.71.05 satisfies it — and 6.0's renderer cleanly skips the Intel
iGPU (`[omni.rtx] Skipping unsupported non-NVIDIA GPU`) and binds the Ada. Verified: the RTX
renderer works on 6.0.1 (offscreen camera tutorial rendered 5987 frames, 0 crashes).

**Isolation:** built in a NEW env `robotlab6` (py3.12) + `~/IsaacLab-3.0` + `~/robot_lab6`. The
working 5.1 stack (`robotlab` env, `~/robot_lab`, trained checkpoint, ONNX, pybullet walk-video
dashboard) is untouched as a fallback.

## Install (see `install_robotlab6.sh`)
- `conda create -n robotlab6 python=3.12` (Isaac Sim 6.X requires 3.12 exactly)
- `uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow`
- `uv pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`
- `git clone https://github.com/isaac-sim/IsaacLab --branch develop ~/IsaacLab-3.0 && ./isaaclab.sh --install`
- robot_lab: fresh clone `~/robot_lab6`, `pip install -e source/robot_lab --no-deps`, copy our D1
  files in (`assets/navbot_d1.py`, `config/quadruped/navbot_d1/*`, `data/Robots/navbot/*`).

## The 9 port fixes (full diff in `robot_lab6_port.diff`)
1. **Blacklist** unneeded task families/robots in `tasks/__init__.py` (`beyondmimic`, `direct`,
   `humanoid`, `wheeled`, `velocity.config.others`, and the 8 other quadruped robots) — they hit
   2.3→3.0 breaks / exotic deps (`cusrl`); we only need `navbot_d1`. (see `tasks__init__.py`)
2. **Stock mdp moved:** `isaaclab_tasks.manager_based.locomotion.velocity.mdp` → `isaaclab_tasks.core.velocity.mdp`.
3. **Noise cfg unified:** `isaaclab.utils.noise.AdditiveUniformNoiseCfg` → `UniformNoiseCfg`
   (default `operation="add"` preserves behavior).
4. **configclass circular-import:** `from isaaclab.utils import configclass` →
   `from isaaclab.utils.configclass import configclass` (the bare form resolved to the *submodule*
   during robot_lab's import walk).
5. **Physics preset refactor:** dropped `sim.physx.gpu_max_rigid_patch_count` (3.0 sets PhysX GPU
   params via multi-backend `PresetCfg`/`PhysxCfg`; default is fine ≤4096 envs).
6. **URDF import prim nesting:** 6.0 imports links under `/Robot/Geometry/<link>` (5.1 was flat
   `/Robot/<link>`) → height-scanner `prim_path` fixed to `{ENV_REGEX_NS}/Robot/Geometry/base`.
7. **Feet merged:** `merge_fixed_joints=True` merged the fixed foot links into the calf on 6.0
   (5.1 kept them) → set `merge_fixed_joints=False` so `.*_foot` contact bodies exist.
8. **Warp data pipeline:** 3.0 returns physics buffers as warp `ProxyArray`; jit-scripted math
   (`quat_apply_inverse`, …) needs the `.torch` view. Added `.torch` to 66 buffer accesses in the
   custom mdp (matching stock `core.velocity.mdp`). (see `patch_torch.py`)
9. (verification tooling: `dump_d1_prims.py`, `test_import.py`)

## Status
- ✅ D1 task registers, env builds, **trains** on Isaac Sim 6.0.1 / Isaac Lab 3.0 (4096 envs).
- RTX viewport confirmed working on 6.0.1 — the native "see it walk" is now possible (no pybullet
  workaround needed).
- TODO: point the Isaac Lab dashboard at `~/robot_lab6/logs`; retrain fully; render in the native
  viewport; revisit the dropped GPU-patch tuning only if a high-env-count run overflows.
