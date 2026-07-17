"""Diagnostic: hold the D1 at its DEFAULT pose with ZERO policy action, and report
whether it stands (base height stays up, body upright, joints at their defaults) or
flops to its belly. Isolates 'default pose / joint convention wrong' vs 'policy is
degenerate'. No trained policy needed."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--out", type=str, default=os.path.expanduser("~/d1_defaultpose.mp4"))
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.observations.policy.enable_corruption = False
    # side follow-cam
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (0.0, 2.2, 0.5)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    if hasattr(env_cfg.commands.base_velocity, "debug_vis"):
        env_cfg.commands.base_velocity.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    robot = env.unwrapped.scene["robot"]

    jn = list(robot.data.joint_names)
    default_jp = robot.data.default_joint_pos.torch[0].detach().cpu().numpy()
    print("[DIAG] joint order:", jn)
    print("[DIAG] default_joint_pos:", np.round(default_jp, 3).tolist())
    lims = robot.data.joint_pos_limits.torch[0].detach().cpu().numpy()
    for i, n in enumerate(jn):
        if "hip" not in n:  # thigh/calf limits reveal any axis flip vs URDF
            print(f"[DIAG] limit {n}: [{lims[i,0]:.3f}, {lims[i,1]:.3f}]  (Frank URDF: thigh[-1.57,2.40] calf[-2.25,0.50])")

    # foot geometry at the default pose (pure FK, before dynamics) — feet should be
    # ~0.43 m BELOW the base for a standing quadruped
    body_names = list(robot.data.body_names)
    foot_ids = [i for i, n in enumerate(body_names) if n.endswith("_foot")]
    bp0 = robot.data.root_pos_w.torch[0].detach().cpu().numpy()
    fp0 = robot.data.body_pos_w.torch[0, foot_ids].detach().cpu().numpy()
    for i, fi in enumerate(foot_ids):
        print(f"[DIAG] foot {body_names[fi]} z relative to base: {fp0[i,2]-bp0[2]:+.3f} m (want ~-0.43)")

    action_dim = env.action_space.shape[-1] if hasattr(env, "action_space") else robot.data.joint_pos.torch.shape[-1]
    zero = torch.zeros((args_cli.num_envs, action_dim), device=env.unwrapped.device)

    frames = []
    env.get_observations()
    heights, grav_z = [], []
    for k in range(args_cli.steps):
        with torch.inference_mode():
            env.step(zero)  # zero action => joints commanded to default pose
        heights.append(float(robot.data.root_pos_w.torch[0, 2]))
        grav_z.append(float(robot.data.projected_gravity_b.torch[0, 2]))
        fr = env.unwrapped.render()
        if fr is not None:
            fr = np.asarray(fr)
            if fr.ndim == 3 and fr.shape[2] >= 3:
                frames.append(fr[:, :, :3].astype(np.uint8))

    settled_jp = robot.data.joint_pos.torch[0].detach().cpu().numpy()
    print(f"[DIAG] base height: start={heights[0]:.3f}  end={heights[-1]:.3f}  min={min(heights):.3f}")
    print(f"[DIAG] projected_gravity_z: start={grav_z[0]:.3f}  end={grav_z[-1]:.3f}  (-1.0 = perfectly upright)")
    print(f"[DIAG] settled joint_pos: {np.round(settled_jp, 3).tolist()}")
    print(f"[DIAG] joint drift from default (max abs): {np.max(np.abs(settled_jp - default_jp)):.3f}")
    if frames:
        imageio.mimwrite(args_cli.out, frames, fps=50, quality=8, macro_block_size=8)
        print(f"[DIAG] wrote {args_cli.out} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
