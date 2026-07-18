"""Render the trained D1 in the NATIVE Isaac Sim 6.0 RTX renderer, capturing
env.render() frames and writing an MP4 with imageio (bypasses the moviepy/
RecordVideo fps issue). Based on robot_lab's play.py policy pipeline."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, default=os.path.expanduser("~/d1_native_walk.mp4"))
parser.add_argument("--vx", type=float, default=0.5, help="fixed forward command m/s")
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
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from importlib import metadata  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

installed_version = metadata.version("rsl-rl-lib")
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.observations.policy.enable_corruption = False
    # FIXED steady forward command (0.5 m/s) so the walk is clean to judge
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    for a, v in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0), ("heading_command", False)):
        if hasattr(cmd, a):
            setattr(cmd, a, v)
    cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(cmd.ranges, "heading"):
        cmd.ranges.heading = (0.0, 0.0)
    if hasattr(cmd, "debug_vis"):
        cmd.debug_vis = False
    # rear-quarter follow camera (behind + side + up) to see the back legs
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-1.7, 1.1, 0.65)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.2)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    env_cfg.log_dir = os.path.dirname(resume)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    jn = list(robot.data.joint_names)
    p0 = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy().copy()
    jp_sum = None
    nacc = 0
    frames = []
    obs = env.get_observations()
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, _, _ = env.step(act)
        jp = robot.data.joint_pos.torch[0].detach().cpu().numpy()
        jp_sum = jp.copy() if jp_sum is None else jp_sum + jp
        nacc += 1
        # drive the render camera to follow env0 explicitly (viewer cfg is
        # ignored on the headless rgb_array path)
        # Follow env0 with the ACTUAL capture camera: Isaac Lab 3.0's rgb_array
        # video path renders /OmniverseKit_Persp via its own Kit capture object
        # (isaaclab_physx IsaacsimKitPerspectiveVideo) — env viewer cfg and
        # sim.set_camera_view do NOT move it. Drive it directly per frame.
        rp = robot.data.root_pos_w.torch[0].detach().cpu().numpy()
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            ViewportManager.set_camera_view(
                "/OmniverseKit_Persp",
                eye=[float(rp[0]) - 1.8, float(rp[1]) + 1.2, float(rp[2]) + 0.7],
                target=[float(rp[0]) + 0.3, float(rp[1]), float(rp[2])],
            )
        except Exception as e:
            if not frames:  # report once, on the first frame
                print(f"[RENDER6] follow-cam FAILED: {type(e).__name__}: {e}")
        if not frames:
            print(f"[RENDER6] env0 root at {np.round(rp, 2).tolist()}")
        fr = env.unwrapped.render()
        if fr is not None:
            fr = np.asarray(fr)
            if fr.ndim == 3 and fr.shape[2] >= 3:
                frames.append(fr[:, :, :3].astype(np.uint8))
    p1 = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy()
    dist = float(((p1 - p0) ** 2).sum() ** 0.5)
    print(f"[RENDER6] env0 traveled {dist:.2f} m in {args_cli.steps} steps ({dist / (args_cli.steps * 0.02):.2f} m/s avg)")
    jpm = jp_sum / max(1, nacc)
    for i, n in enumerate(jn):
        if "thigh" in n or "calf" in n:
            print(f"[JOINT] {n}: mean {jpm[i]:+.3f} rad  (default -0.75; near -0.75 = leg down/back, toward 0 = horizontal)")

    if frames:
        imageio.mimwrite(args_cli.out, frames, fps=50, quality=8, macro_block_size=8)
        print(f"[RENDER6] wrote {args_cli.out} ({len(frames)} frames, {frames[0].shape})")
    else:
        print("[RENDER6] no frames captured")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
