"""Spawn the D1 and print its imported body/link names + prim paths (6.0 URDF import)."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
simulation_app = AppLauncher(args_cli).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from robot_lab.assets.navbot_d1 import NAVBOT_D1_CFG  # noqa: E402

sim = SimulationContext(SimulationCfg(dt=0.005))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = Articulation(NAVBOT_D1_CFG.replace(prim_path="/World/Robot"))
sim.reset()
print("BODY_NAMES:", list(robot.data.body_names))
print("ROOT_BODY:", robot.data.body_names[0] if len(robot.data.body_names) else "?")

import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()
paths = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith("/World/Robot")]
print("N_PATHS:", len(paths))
print("BASE_PATHS:", [p for p in paths if p.rstrip("/").endswith("base")])
print("NON_PHYSICS_PATHS:", [p for p in paths if "/Physics" not in p][:20])
simulation_app.close()
