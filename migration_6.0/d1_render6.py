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
    # turn off the velocity-command debug marker so it doesn't occlude the robot
    if hasattr(env_cfg.commands.base_velocity, "debug_vis"):
        env_cfg.commands.base_velocity.debug_vis = False
    # side-low follow camera (relative to robot frame) so the leg motion is visible
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.4, 1.4, 0.55)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

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
    p0 = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy().copy()
    frames = []
    obs = env.get_observations()
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            act = policy(obs)
            obs, _, _, _ = env.step(act)
        fr = env.unwrapped.render()
        if fr is not None:
            fr = np.asarray(fr)
            if fr.ndim == 3 and fr.shape[2] >= 3:
                frames.append(fr[:, :, :3].astype(np.uint8))
    p1 = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy()
    dist = float(((p1 - p0) ** 2).sum() ** 0.5)
    print(f"[RENDER6] env0 traveled {dist:.2f} m in {args_cli.steps} steps ({dist / (args_cli.steps * 0.02):.2f} m/s avg)")

    if frames:
        imageio.mimwrite(args_cli.out, frames, fps=50, quality=8, macro_block_size=8)
        print(f"[RENDER6] wrote {args_cli.out} ({len(frames)} frames, {frames[0].shape})")
    else:
        print("[RENDER6] no frames captured")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
