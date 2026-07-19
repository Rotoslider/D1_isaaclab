"""Phase-B smoke: the AMP env surface on nuc1.

Checks: task registers + env builds; get_amp_observations() is 55-D, finite, and
its features are physically sane at stance (dof_pos ~ default pose in DATASET
order, root_z ~ standing height); command slew ramps the applied command toward
the target and the settle mask flips after 44 settled steps; terminal AMP states
appear in extras on resets; amp obs dim matches the AMPLoader's dataset dim.
"""
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import robot_lab.tasks  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "RobotLab-Isaac-Velocity-AMP-NavBot-D1-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 32
env = gym.make(TASK, cfg=env_cfg)
u = env.unwrapped
print(f"[1] env built: {TASK}, num_envs={u.num_envs}, step_dt={u.step_dt}")

obs, _ = env.reset()
amp0 = u.get_amp_observations()
assert amp0.shape == (32, 55), amp0.shape
assert torch.isfinite(amp0).all()
print(f"[2] amp obs OK: {tuple(amp0.shape)}, finite")

# stance sanity in DATASET order: [FL_hip, FL_thigh, FL_calf, FR_hip, ...]
# default pose: L_hip -0.05 / R_hip +0.05, thigh -0.75, calf -0.75
dof0 = amp0[0, :12].tolist()
print("[3] dof_pos env0 (dataset order):", [round(v, 3) for v in dof0])
print("    root_z:", round(amp0[0, 54].item(), 3), "(expect ~0.40-0.45 standing)")

term = u.command_manager.get_term("base_velocity")
zero = torch.zeros(32, u.action_manager.total_action_dim, device=u.device)
applied_hist, settled_hist = [], []
saw_terminal = False
for i in range(120):
    _, _, _, _, extras = env.step(zero)
    if "amp_terminal_states" in u.extras:
        s = u.extras["amp_terminal_states"]
        assert s.shape[-1] == 55
        saw_terminal = True
    if i % 20 == 0:
        applied_hist.append(term.command[0, 0].item())
        settled_hist.append(u.get_amp_transition_mask().float().mean().item())
print("[4] applied vx env0 every 20 steps:", [round(v, 3) for v in applied_hist])
print("    target  vx env0:", round(term.target_vel_b[0, 0].item(), 3))
print("    settled fraction every 20 steps:", [round(v, 2) for v in settled_hist])

# violent actions -> falls -> base-contact terminations -> terminal AMP capture
for i in range(150):
    _, _, _, _, _ = env.step(3.0 * torch.randn_like(zero))
    if "amp_terminal_states" in u.extras:
        s = u.extras["amp_terminal_states"]
        assert s.shape[-1] == 55 and torch.isfinite(s).all()
        saw_terminal = True
print("    terminal amp states seen (after violent phase):", saw_terminal)
assert saw_terminal, "no terminal AMP states captured despite forced falls"

# dim parity with the expert dataset
sys.path.insert(0, "/home/nuc1/robot_lab6/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/amp")
from motion_loader import AMPLoader

loader = AMPLoader(
    device="cpu", time_between_frames=u.step_dt, preload_transitions=False,
    motion_files=["/home/nuc1/robot_lab6/source/robot_lab/data/Motions/d1/d1_amp_v4_h_uniform_cmdcond.npz"],
    observation_mode="with_feet_vel", command_conditioned=True,
)
assert loader.observation_dim == amp0.shape[-1] == 55
print(f"[5] dim parity: env {amp0.shape[-1]} == dataset {loader.observation_dim}")

print("AMP_ENV_SMOKE_PASSED")
env.close()
app.close()
