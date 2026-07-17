"""Launch the app first (like train.py/play.py do), THEN import robot_lab.tasks."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import robot_lab.tasks  # noqa: F401, E402

navbot = [e for e in gym.registry if "NavBot" in e]
print("REGISTERED_NAVBOT:", navbot)
simulation_app.close()
