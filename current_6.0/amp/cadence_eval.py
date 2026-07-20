"""Cadence eval: policy foot-touchdown frequency vs reference-clip stride frequency.

Policy side: headless rollout at fixed vx; touchdown = rising edge of foot contact
(Fz > 1 N). Cadence = touchdowns per foot per second (trot: 1 touchdown per foot
per stride cycle).
Expert side: per matching forward clip in the npz, stride freq from toe-height
oscillation (mean upward zero-crossings of toe_z - mean over the clip) at 50 Hz.
Usage: python cadence_eval.py <vx> <load_run>
"""
import sys

vx, load_run = float(sys.argv[1]), sys.argv[2]

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import robot_lab.tasks  # noqa: F401
from importlib import metadata as _md
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

TASK = "RobotLab-Isaac-Velocity-AMP-NavBot-D1-v0"
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
N, STEPS, DT = 16, 400, 0.02

env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
env_cfg.scene.num_envs = N
cmd = env_cfg.commands.base_velocity
cmd.resampling_time_range = (1.0e9, 1.0e9)
cmd.rel_standing_envs = 0.0
cmd.ranges.lin_vel_x = (vx, vx)
cmd.ranges.lin_vel_y = (0.0, 0.0)
cmd.ranges.ang_vel_z = (0.0, 0.0)

env = gym.make(TASK, cfg=env_cfg)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
import glob as _glob
ckpts = _glob.glob(f"logs/rsl_rl/navbot_d1_amp/{load_run}/model_*.pt")
resume = max(ckpts, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
print(f"[CADENCE] checkpoint: {resume}")
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume)
policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

sensor = env.unwrapped.scene["contact_forces"]
foot_ids = [sensor.body_names.index(f) for f in FEET]
obs = wrapped.get_observations()
prev_contact = None
touchdowns = torch.zeros(N, len(FEET), device=env.unwrapped.device)
counted = 0
for i in range(STEPS):
    with torch.inference_mode():
        obs, _, _, _ = wrapped.step(policy(obs))
    fz = sensor.data.net_forces_w.torch[:, foot_ids, 2].abs()
    contact = fz > 1.0
    if prev_contact is not None and i >= 60:  # skip command slew-up transient
        touchdowns += (contact & ~prev_contact).float()
        counted += 1
    prev_contact = contact.clone()

per_foot_hz = touchdowns / (counted * DT)
print(f"[CADENCE] policy @vx={vx}: per-foot touchdown Hz mean {per_foot_hz.mean():.2f} "
      f"(per-foot {['%.2f' % v for v in per_foot_hz.mean(dim=0).tolist()]}, "
      f"env spread {per_foot_hz.mean(dim=1).min():.2f}-{per_foot_hz.mean(dim=1).max():.2f})")

# expert reference: forward clips near this vx
d = np.load("/home/nuc1/robot_lab6/source/robot_lab/data/Motions/d1/d1_amp_v4_h_uniform_cmdcond.npz",
            allow_pickle=True)
frames, lens, names, cmds = d["frames"], d["sequence_lengths"], d["sequence_names"], d["commands"]
layout = list(d["frame_layout"])
toe_z_cols = [i for i, c in enumerate(layout) if "toe" in c.lower() and c.endswith("_z") and "vel" not in c.lower()]
if not toe_z_cols:  # fall back: toe_pos block is cols 19..30 (x,y,z per foot), z = 21,24,27,30
    toe_z_cols = [21, 24, 27, 30]
off = 0
print(f"[CADENCE] expert forward clips near vx={vx} (toe-z oscillation, 50 Hz):")
for k in range(len(lens)):
    clip = frames[off:off + lens[k]]
    off += lens[k]
    cvx, cvy, cwz = cmds[k]
    if abs(cvy) > 0.05 or abs(cwz) > 0.05 or not (vx - 0.16 <= cvx <= vx + 0.16) or cvx <= 0:
        continue
    hz = []
    for c in toe_z_cols:
        z = clip[:, c] - clip[:, c].mean()
        crossings = np.sum((z[:-1] < 0) & (z[1:] >= 0))
        hz.append(crossings / (len(clip) / 50.0))
    print(f"    {names[k]} (cmd vx={cvx:.2f}): per-foot stride Hz mean {np.mean(hz):.2f}")

env.close()
app.close()
