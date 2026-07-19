# Frank's cmdcond command machinery (D1_HIMLoco d1.py:452-563) as an Isaac Lab
# command term: the sampled command is the TARGET; the command every consumer
# (obs, rewards, metrics) reads is the APPLIED command, slewed toward the target
# with per-axis accel/decel and projected onto the deployment Stage-0 envelope.
# AMP discriminator transitions are gated on `amp_transition_mask` — True only
# after the applied command has sat within `settle_tolerance` of the target for
# `settle_seconds` (Frank: 0.88 s = 44 steps at dt 0.02).
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass

from .command_smoothing import project_d1_stage0_command_envelope, slew_toward_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SlewedVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand whose published command slews toward the sampled target."""

    cfg: SlewedVelocityCommandCfg

    def __init__(self, cfg: SlewedVelocityCommandCfg, env: ManagerBasedEnv):
        if cfg.heading_command:
            raise ValueError("SlewedVelocityCommand does not support heading commands.")
        super().__init__(cfg, env)
        self.target_vel_b = torch.zeros_like(self.vel_command_b)
        self.command_settled_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._accel = torch.tensor(cfg.accel, dtype=torch.float, device=self.device).unsqueeze(0)
        self._decel = torch.tensor(cfg.decel, dtype=torch.float, device=self.device).unsqueeze(0)

    @property
    def settle_required_steps(self) -> int:
        dt = max(float(self._env.step_dt), 1.0e-6)
        return max(1, int(math.ceil(max(float(self.cfg.settle_seconds), 0.0) / dt)))

    @property
    def amp_transition_mask(self) -> torch.Tensor:
        """True where the applied command has been at its target long enough for AMP."""
        return self.command_settled_steps >= self.settle_required_steps

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        # Fresh episodes start from rest (Frank resets to nominal pose with
        # exact_rest_reset_prob=0.75): applied command begins at zero and slews up.
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # Frank _finalize_resampled_commands: the freshly sampled values become the
        # target; the published command carries the previous applied value forward.
        previous_applied = self.vel_command_b[env_ids].clone()
        super()._resample_command(env_ids)
        target = self.vel_command_b[env_ids].clone()
        if self.cfg.project_stage0_envelope:
            # Frank samples targets from bucketed anchors that live INSIDE the
            # deployment envelope; our uniform box sampling does not, and an
            # out-of-envelope target can never be reached by the projected
            # applied command — the env would never settle and produce no AMP
            # transitions. Projecting the target restores reachability.
            target = project_d1_stage0_command_envelope(target)
        target[self.is_standing_env[env_ids]] = 0.0
        self.target_vel_b[env_ids] = target
        self.vel_command_b[env_ids] = previous_applied
        tolerance = max(float(self.cfg.settle_tolerance), 0.0)
        unchanged = (target - previous_applied).abs().max(dim=1).values <= tolerance
        required = self.settle_required_steps
        self.command_settled_steps[env_ids] = torch.where(
            unchanged,
            torch.full_like(self.command_settled_steps[env_ids], required),
            torch.zeros_like(self.command_settled_steps[env_ids]),
        )

    def _update_command(self):
        # (no super() call: the base zeroes vel_command_b for standing envs directly,
        # which would bypass the slew — standing envs zero the TARGET instead)
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.target_vel_b[standing_ids] = 0.0
        self.vel_command_b[:] = slew_toward_tensor(
            self.vel_command_b, self.target_vel_b, self._accel, self._decel, float(self._env.step_dt)
        )
        if self.cfg.project_stage0_envelope:
            self.vel_command_b[:] = project_d1_stage0_command_envelope(self.vel_command_b)
        tolerance = max(float(self.cfg.settle_tolerance), 0.0)
        close = (self.target_vel_b - self.vel_command_b).abs().max(dim=1).values <= tolerance
        required = self.settle_required_steps
        self.command_settled_steps = torch.where(
            close,
            torch.clamp(self.command_settled_steps + 1, max=required),
            torch.zeros_like(self.command_settled_steps),
        )


@configclass
class SlewedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`SlewedVelocityCommand` (Frank cmdcond defaults)."""

    class_type: type = SlewedVelocityCommand

    accel: tuple[float, float, float] = (1.6, 1.2, 2.0)
    """Per-axis (x, y, yaw) slew rate away from zero [unit/s] (command_accel_*)."""
    decel: tuple[float, float, float] = (2.4, 1.8, 3.0)
    """Per-axis slew rate toward zero / reducing magnitude [unit/s] (command_decel_*)."""
    settle_tolerance: float = 0.02
    """Max |target - applied| on any axis to count as settled (amp_transition_command_tolerance)."""
    settle_seconds: float = 0.88
    """Time the command must stay settled before AMP transitions are valid."""
    project_stage0_envelope: bool = True
    """Project the applied command onto the rl_sar Stage-0 deployment envelope."""
