"""Smoke test: Frank's vendored AMP plumbing on nuc1 (robotlab6 env, Blackwell).

Verifies, with the canonical cmdcond settings from d1_config.py:
  1. AMPLoader loads the 92-clip himloco_amp_npz_v1 dataset, observation_dim=55
  2. command-conditioned transition sampling (s, s_next, cmd)
  3. AMPDiscriminator forward (input 55*2+3=113) + style reward math
  4. Normalizer + ReplayBuffer round-trip
No Isaac Sim involved — pure torch on cuda:0.
"""
import sys, torch

AMP_PKG = "/home/nuc1/robot_lab6/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/amp"
sys.path.insert(0, AMP_PKG)
from motion_loader import AMPLoader
from amp_discriminator import AMPDiscriminator
from amp_utils import Normalizer
from replay_buffer import ReplayBuffer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
NPZ = "/home/nuc1/robot_lab6/source/robot_lab/data/Motions/d1/d1_amp_v4_h_uniform_cmdcond.npz"

# canonical cmdcond settings (D1AMPCanonicalCmdCondCfgPPO)
loader = AMPLoader(
    device=device,
    time_between_frames=0.02,           # env.dt, matches 50 Hz dataset
    preload_transitions=True,
    num_preload_transitions=200_000,    # 2M in the real run; smaller for smoke
    motion_files=[NPZ],
    observation_mode="with_feet_vel",
    reorder_from_pybullet_to_isaac=False,
    command_conditioned=True,
    command_stage_envelopes=[dict(start_iteration=0, vx_min=-0.60, vx_max=0.90, vy_abs_max=0.40, wz_abs_max=0.50)],
)
obs_dim = loader.observation_dim
n_clips = len(loader.trajectory_lens) if hasattr(loader, "trajectory_lens") else "?"
print(f"[1] loader OK: observation_dim={obs_dim} (expect 55), clips={n_clips}")
assert obs_dim == 55

gen = loader.feed_forward_generator(num_mini_batch=1, mini_batch_size=4096)
batch = next(iter(gen))
s, s_next, cmd = batch
print(f"[2] expert batch OK: s{tuple(s.shape)} s_next{tuple(s_next.shape)} cmd{tuple(cmd.shape)}")
assert s.shape == (4096, 55) and cmd.shape == (4096, 3)
assert torch.isfinite(s).all() and torch.isfinite(s_next).all() and torch.isfinite(cmd).all()

disc = AMPDiscriminator(
    input_dim=obs_dim * 2,
    amp_reward_coef=0.08,
    hidden_layer_sizes=[1024, 512],
    device=device,
    task_reward_lerp=0.75,
    command_dim=3,
).to(device)
norm = Normalizer(obs_dim)
task_r = torch.rand(4096, device=device)
with torch.no_grad():
    d = disc.amp_linear(disc.trunk(disc._assemble_input(s, s_next, cmd)))
    reward, raw_d = disc.predict_amp_reward(s, s_next, task_r, normalizer=norm, command=cmd)
style = 0.08 * torch.clamp(1 - 0.25 * (raw_d.squeeze() - 1) ** 2, min=0)
expect = 0.25 * style + 0.75 * task_r
print(f"[3] disc OK: logits{tuple(d.shape)} mean={d.mean():.4f}; blended reward mean={reward.mean():.4f}")
assert torch.allclose(reward, expect, atol=1e-5), "style/task lerp mismatch vs hand-computed"

rb = ReplayBuffer(obs_dim, 100_000, device, command_dim=3)
rb.insert(s, s_next, cmd)
pol = next(rb.feed_forward_generator(num_mini_batch=1, mini_batch_size=1024))
print(f"[4] replay buffer OK: sampled {tuple(pol[0].shape)} (+cmd {tuple(pol[2].shape)})")

print("AMP_PLUMBING_SMOKE_PASSED")
