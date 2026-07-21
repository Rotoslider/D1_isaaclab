"""Interactive Isaac Sim viewer for trained D1 runs — the window equivalent of
d1_render6.py's videos: steady forward command by default so the gait is judgeable,
--random for varied (but sane) commands. Exits when the window is closed.

Run from ~/robot_lab6 with --viz kit for the GUI window:
    python scripts/reinforcement_learning/rsl_rl/d1_view.py --load_run <run> --viz kit [--vx 0.6] [--random]
"""
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
parser.add_argument("--vx", type=float, default=0.6, help="fixed forward command m/s (ignored with --random)")
parser.add_argument("--random", action="store_true", help="varied commands (sane ranges) instead of fixed forward")
parser.add_argument("--test_steps", type=int, default=0, help="internal: step N times then exit (headless smoke)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
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

    cmd = env_cfg.commands.base_velocity
    if args_cli.random:
        # varied but within trained competence (raw cfg ranges include the
        # curriculum cap +-1.6 and +-3.14 yaw -> looks like chaos in a viewer)
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-0.5, 0.5)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.rel_standing_envs = 0.05
    else:
        cmd.resampling_time_range = (1.0e9, 1.0e9)
        for a, v in (("rel_standing_envs", 0.0), ("rel_heading_envs", 0.0), ("heading_command", False)):
            if hasattr(cmd, a):
                setattr(cmd, a, v)
        cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(cmd.ranges, "heading"):
            cmd.ranges.heading = (0.0, 0.0)

    # follow env0 with the interactive viewport camera (user can still orbit)
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-2.0, 1.4, 0.8)
    env_cfg.viewer.lookat = (0.4, 0.0, 0.2)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    if agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        # OnPolicyRunner handles inference for PPO AND AmpPPO checkpoints (the
        # AmpOnPolicyRunner class only differs in training; extra disc keys ignored)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[VIEW] policy: {resume}")
    print(f"[VIEW] commands: {'random (sane ranges)' if args_cli.random else f'fixed forward {args_cli.vx} m/s'}")
    print("[VIEW] close the Isaac Sim window to exit.")

    obs = env.get_observations()
    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        steps += 1
        if args_cli.test_steps and steps >= args_cli.test_steps:
            print(f"[VIEW] test mode: {steps} steps OK, exiting")
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
