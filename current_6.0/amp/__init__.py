# AMP (Adversarial Motion Priors) port of Frank's D1_HIMLoco canonical cmdcond
# stack (commit a88b463) onto Isaac Lab 3.0 / rsl-rl-lib.
#
# Vendored verbatim from the D1_HIMLoco vendored rsl_rl (pure torch/numpy, no
# Isaac Gym deps):
#   amp_discriminator.py — LSGAN discriminator, style reward
#     0.08*clamp(1-0.25(d-1)^2), task/style lerp, zero-centered grad penalty,
#     3-D command conditioning (input 55*2+3=113)
#   motion_loader.py     — AMPLoader for himloco_amp_npz_v1 datasets (61-D
#     frames -> 55-D "with_feet_vel" AMP obs), preloaded transition pairs,
#     command-conditioned expert sampling
#   replay_buffer.py     — circular policy-transition buffer (state, next_state,
#     command)
#   amp_utils.py         — RunningMeanStd/Normalizer (eps=1e-4, clip=10)
#
# Isaac-Lab-side integration (runner/algorithm subclasses, env AMP surface,
# command slew/settle) lives alongside these; see AMP_PORT_DESIGN.md in the
# isaaclab_port repo.

from .amp_discriminator import AMPDiscriminator
from .amp_utils import Normalizer, RunningMeanStd
from .motion_loader import AMPLoader
from .replay_buffer import ReplayBuffer

# Isaac-Lab-side integration (import AmpVelocityEnv / AmpOnPolicyRunner / AmpPPO
# from their modules directly — amp_env pulls in isaaclab, amp_runner pulls in
# rsl_rl; keeping them out of this namespace lets the vendored core stay
# importable in a bare python env, e.g. for dataset tooling).
