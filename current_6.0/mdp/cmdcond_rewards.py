# Frank's D1AMPCanonicalCmdCondCfg reward functions (D1_HIMLoco / legged_gym) ported to
# Isaac Lab's manager-based reward API — Stage 1 of the cmdcond alignment
# (CMDCOND_REWARD_MAP.md sections A + B + C + moving_calf_contact).
# Formulas match legged_gym/envs/d1/d1.py exactly; only the data access is Isaac-Lab-native.
#
# Command layout matches legged_gym: command[:,0]=lin_vel_x, [:,1]=lin_vel_y, [:,2]=yaw.
# Hip terms REQUIRE asset_cfg joint_names ordered [FL, FR, RL, RR] with preserve_order=True:
# the HAA outward signs are [+1, -1, +1, -1] (d1.py:603 `_get_haa_outward_signs`, dof order
# FL,FR,RL,RR) and the targets are [front, front, rear, rear].
# Where Frank reads a commanded joint target (hip_target_inward_limit) we use the actual
# joint_pos — the PD-tracked proxy used throughout frank_rewards.py.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor

from .frank_rewards import _cmd, _contact_z, _moving_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "has_contact",
    "stuck",
    "hip_outward_weak_inward",
    "hip_target_inward_limit",
    "zero_action_rate",
    "zero_dof_vel",
    "zero_feet_vel",
    "zero_base_vel_z",
    "zero_yaw_rate",
    "command_xy_speed_shortfall",
    "command_yaw_rate_shortfall",
    "lateral_speed_shortfall",
    "diagonal_component_speed_shortfall",
    "SmoothnessSq",
    "moving_calf_contact",
]

# HAA outward signs for hips ordered [FL, FR, RL, RR] (d1.py:603): left +1, right -1.
_HAA_OUTWARD_SIGNS = [1.0, -1.0, 1.0, -1.0]


def _zero_cmd_mask(env, command_name: str) -> torch.Tensor:
    """Frank._zero_cmd_mask analog form (d1.py:675): |lin_cmd|<0.06 & |yaw_cmd|<0.06."""
    c = _cmd(env, command_name)
    lin_cmd = torch.norm(c[:, :2], dim=1)
    yaw_cmd = torch.abs(c[:, 2])
    return ((lin_cmd < 0.06) & (yaw_cmd < 0.06)).float()


# --------------------------------------------------------------------------- #
# A. core terms missing from the frank port                                    #
# --------------------------------------------------------------------------- #
def has_contact(
    env, command_name: str, sensor_cfg: SceneEntityCfg, force_thresh: float = 1.0
) -> torch.Tensor:
    """Frank._reward_has_contact (d1.py:3055): while standing (~moving(0.1, 0.05)), reward the
    fraction of feet in contact (Fz > 1.0, contact_filt OR-filter ~ sensor history max)."""
    standing = ~_moving_mask(env, command_name, 0.1, 0.05)
    contact = _contact_z(env, sensor_cfg, force_thresh).float()
    return standing.float() * torch.sum(contact, dim=-1) / contact.shape[1]


def stuck(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Frank._reward_stuck (d1.py:3078): commanded to move but barely moving."""
    asset: Articulation = env.scene[asset_cfg.name]
    small_lin_vel = torch.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1) < 0.1
    small_ang_vel = torch.abs(asset.data.root_ang_vel_b.torch[:, 2]) < 0.1
    large_commands = _moving_mask(env, command_name, 0.1, 0.1)
    return (small_lin_vel & small_ang_vel & large_commands).float()


def hip_outward_weak_inward(
    env,
    asset_cfg: SceneEntityCfg,
    target_front: float = -0.055,
    target_rear: float = -0.065,
    margin: float = 0.05,
) -> torch.Tensor:
    """Frank._reward_hip_outward_weak_inward (d1.py:3128): pull hip abduction toward a slightly
    inward target. asset_cfg joints MUST be [FL, FR, RL, RR] hips with preserve_order=True."""
    asset: Articulation = env.scene[asset_cfg.name]
    q_hip = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    signs = torch.tensor(_HAA_OUTWARD_SIGNS, device=q_hip.device, dtype=q_hip.dtype).unsqueeze(0)
    target = torch.tensor(
        [target_front, target_front, target_rear, target_rear], device=q_hip.device, dtype=q_hip.dtype
    ).unsqueeze(0)
    outward = q_hip * signs
    err = (outward - target) / max(margin, 1e-6)
    return torch.mean(torch.square(err), dim=1)


def hip_target_inward_limit(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    limit_front: float = -0.090,
    limit_rear: float = -0.100,
    margin: float = 0.04,
) -> torch.Tensor:
    """Frank._reward_hip_target_inward_limit (d1.py:3142): while moving (0.1, 0.2), penalize hip
    abduction crossing an inward limit. Uses actual hip joint_pos (PD-tracked proxy for Frank's
    joint_pos_target). asset_cfg joints MUST be [FL, FR, RL, RR] hips with preserve_order=True."""
    asset: Articulation = env.scene[asset_cfg.name]
    q_hip = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    signs = torch.tensor(_HAA_OUTWARD_SIGNS, device=q_hip.device, dtype=q_hip.dtype).unsqueeze(0)
    limit = torch.tensor(
        [limit_front, limit_front, limit_rear, limit_rear], device=q_hip.device, dtype=q_hip.dtype
    ).unsqueeze(0)
    outward = q_hip * signs
    err = torch.clamp(limit - outward, min=0.0) / max(margin, 1e-6)
    moving = _moving_mask(env, command_name, 0.10, 0.20)
    return torch.mean(torch.square(err), dim=1) * moving.float()


# --------------------------------------------------------------------------- #
# B. zero-command hold (all masked by the analog zero_cmd mask)               #
# --------------------------------------------------------------------------- #
def zero_action_rate(env, command_name: str) -> torch.Tensor:
    """Frank._reward_zero_action_rate (d1.py:3113): sum((a - a_prev)^2) on zero command."""
    diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(diff), dim=1) * _zero_cmd_mask(env, command_name)


def zero_dof_vel(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Frank._reward_zero_dof_vel (d1.py:3116): sum(dof_vel^2) on zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_sq = torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)
    return vel_sq * _zero_cmd_mask(env, command_name)


def zero_feet_vel(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Frank._reward_zero_feet_vel (d1.py:3119): sum(feet_vel^2) (world frame, all axes) on
    zero command. Pass asset_cfg with the foot body_names."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(feet_vel), dim=(1, 2)) * _zero_cmd_mask(env, command_name)


def zero_base_vel_z(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Frank._reward_zero_base_vel_z (d1.py:3122): base-frame vz^2 on zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 2]) * _zero_cmd_mask(env, command_name)


def zero_yaw_rate(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Frank._reward_zero_yaw_rate (d1.py:3125): base-frame yaw rate^2 on zero command."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b.torch[:, 2]) * _zero_cmd_mask(env, command_name)


# --------------------------------------------------------------------------- #
# C. command-shortfall terms                                                   #
# --------------------------------------------------------------------------- #
def command_xy_speed_shortfall(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_min: float = 0.075,
    margin: float = 0.015,
    denom_bias: float = 0.10,
    err_cap: float = 2.0,
) -> torch.Tensor:
    """Frank._reward_command_xy_speed_shortfall (d1.py:3622): penalize only underspeed along
    the commanded planar direction (base frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_xy = _cmd(env, command_name)[:, :2]
    cmd_speed = torch.norm(cmd_xy, dim=1)
    cmd_dir = cmd_xy / torch.clamp(cmd_speed, min=1e-6).unsqueeze(1)
    aligned_speed = torch.sum(asset.data.root_lin_vel_b.torch[:, :2] * cmd_dir, dim=1)
    shortfall = torch.clamp(cmd_speed - aligned_speed - margin, min=0.0)
    err = shortfall / torch.clamp(cmd_speed + denom_bias, min=1e-6)
    return torch.square(torch.clamp(err, max=err_cap)) * (cmd_speed >= cmd_min).float()


def command_yaw_rate_shortfall(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_min: float = 0.10,
    margin: float = 0.03,
    denom_bias: float = 0.15,
    err_cap: float = 2.0,
) -> torch.Tensor:
    """Frank._reward_command_yaw_rate_shortfall (d1.py:3637): penalize only yaw underspeed in
    the commanded direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_yaw = _cmd(env, command_name)[:, 2]
    cmd_abs = torch.abs(cmd_yaw)
    aligned_yaw = asset.data.root_ang_vel_b.torch[:, 2] * torch.sign(cmd_yaw)
    shortfall = torch.clamp(cmd_abs - aligned_yaw - margin, min=0.0)
    err = shortfall / torch.clamp(cmd_abs + denom_bias, min=1e-6)
    return torch.square(torch.clamp(err, max=err_cap)) * (cmd_abs >= cmd_min).float()


def lateral_speed_shortfall(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_min: float = 0.075,
    margin: float = 0.03,
    forward_max: float = 0.08,
    yaw_max: float = 0.10,
) -> torch.Tensor:
    """Frank._reward_lateral_speed_shortfall (d1.py:3651): on near-pure lateral commands,
    penalize underspeed along the commanded y direction (unnormalized square)."""
    asset: Articulation = env.scene[asset_cfg.name]
    c = _cmd(env, command_name)
    cmd_y = c[:, 1]
    cmd_abs = torch.abs(cmd_y)
    lateral = (cmd_abs > cmd_min) & (torch.abs(c[:, 0]) <= forward_max) & (torch.abs(c[:, 2]) <= yaw_max)
    aligned_lateral_vel = asset.data.root_lin_vel_b.torch[:, 1] * torch.sign(cmd_y)
    shortfall = torch.clamp(cmd_abs - aligned_lateral_vel - margin, min=0.0)
    return torch.square(shortfall) * lateral.float()


def diagonal_component_speed_shortfall(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cmd_min: float = 0.075,
    margin: float = 0.03,
    denom_bias: float = 0.10,
    err_cap: float = 2.0,
    yaw_max: float = 0.10,
) -> torch.Tensor:
    """Frank._reward_diagonal_component_speed_shortfall (d1.py:3667): on diagonal commands,
    penalize vx/vy underspeed independently so both components stay observable."""
    asset: Articulation = env.scene[asset_cfg.name]
    c = _cmd(env, command_name)
    cmd = c[:, :2]
    cmd_abs = torch.abs(cmd)
    diagonal = (cmd_abs[:, 0] >= cmd_min) & (cmd_abs[:, 1] >= cmd_min) & (torch.abs(c[:, 2]) <= yaw_max)
    aligned = asset.data.root_lin_vel_b.torch[:, :2] * torch.sign(cmd)
    shortfall = torch.clamp(cmd_abs - aligned - margin, min=0.0)
    normalized = shortfall / torch.clamp(cmd_abs + denom_bias, min=1e-6)
    return torch.sum(torch.square(torch.clamp(normalized, max=err_cap)), dim=1) * diagonal.float()


# --------------------------------------------------------------------------- #
# square-form smoothness + Stage-1 slice of section D                          #
# --------------------------------------------------------------------------- #
class SmoothnessSq(ManagerTermBase):
    """Frank cmdcond _reward_smoothness (d1.py:3015): sum((a - 2*a_prev + a_prev_prev)^2).
    Square form — replaces frank_rewards.Smoothness (sqrt form) for the cmdcond alignment."""

    def __init__(self, cfg: RewTerm, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n = env.action_manager.total_action_dim
        self.last = torch.zeros(env.num_envs, n, device=env.device)
        self.last_last = torch.zeros(env.num_envs, n, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.last[env_ids] = 0.0
        self.last_last[env_ids] = 0.0

    def __call__(self, env: "ManagerBasedRLEnv") -> torch.Tensor:
        act = env.action_manager.action
        jerk = act - 2.0 * self.last + self.last_last
        rew = torch.sum(torch.square(jerk), dim=1)
        self.last_last = self.last.clone()
        self.last = act.clone()
        return rew


def moving_calf_contact(
    env, command_name: str, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Frank._reward_moving_calf_contact (d1.py:4032): any calf contact (|F| > threshold) while
    commanded to move (0.1, 0.1). Pass sensor_cfg with the calf body_names."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    calf_contact = torch.any(torch.norm(forces, dim=-1).max(dim=1)[0] > force_threshold, dim=1)
    moving = _moving_mask(env, command_name, 0.1, 0.1)
    return calf_contact.float() * moving.float()
