# Frank's D1RoughCfg reward functions (from D1_HIMLoco / legged_gym) ported to
# Isaac Lab's manager-based reward API. Formulas match legged_gym/envs/d1/d1.py
# and envs/base/legged_robot.py exactly; only the data access is Isaac-Lab-native.
#
# Command layout matches legged_gym: command[:,0]=lin_vel_x, [:,1]=lin_vel_y, [:,2]=yaw.
# Hip / foot subsets are resolved BY NAME via SceneEntityCfg(joint_names/body_names),
# so Isaac Lab's joint ordering (which differs from legged_gym's [0,3,6,9]) is handled
# correctly. Where Frank reads a commanded joint target we use the actual joint_pos
# (PD-tracked, functionally equivalent and far more robust across the port).
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "lin_vel_z_frank",
    "ang_vel_xy_frank",
    "action_rate_frank",
    "dof_vel_frank",
    "foot_slip_frank",
    "stand_still_frank",
    "joint_pos_penalty_frank",
    "zero_hip_target_dev",
    "base_height_band",
    "forward_min_contact_count",
    "forward_base_vertical_velocity",
    "torque_saturation",
    "Smoothness",
    "TouchdownImpact",
    "ForwardSwingPeaks",
]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _cmd(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _moving_mask(env, command_name, lin_thresh=0.1, yaw_thresh=0.1) -> torch.Tensor:
    """Frank._command_moving_mask: True where the robot is commanded to move."""
    c = _cmd(env, command_name)
    lin = torch.norm(c[:, :2], dim=1)
    yaw = torch.abs(c[:, 2])
    return torch.logical_or(lin > lin_thresh, yaw > yaw_thresh)


def _forward_mask(env, command_name, cmd_min=0.3, lat_max=0.1, yaw_max=0.2) -> torch.Tensor:
    """Frank._forward_straight_mask: True on a near-straight forward command."""
    c = _cmd(env, command_name)
    return (c[:, 0] > cmd_min) & (torch.abs(c[:, 1]) < lat_max) & (torch.abs(c[:, 2]) < yaw_max)


def _contact_z(env, sensor_cfg: SceneEntityCfg, thresh: float) -> torch.Tensor:
    """Per-foot contact bool from the max upward contact force over the history buffer."""
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].max(dim=1)[0]
    return fz > thresh


def _base_height(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terrain-relative base height (Frank._get_base_heights) via the base RayCaster."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_z = sensor.data.ray_hits_w[..., 2]
    if torch.isnan(ray_z).any() or torch.isinf(ray_z).any() or torch.max(torch.abs(ray_z)) > 1e6:
        return asset.data.root_link_pos_w.torch[:, 2]
    return asset.data.root_link_pos_w.torch[:, 2] - torch.mean(ray_z, dim=1)


def _feet_height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Approx terrain-relative foot height = world foot z minus the env terrain origin z.
    Exact on flat terrain (where the forward gait is shaped); approximate on rough."""
    asset: Articulation = env.scene[asset_cfg.name]
    fz = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return fz - env.scene.env_origins[:, 2].unsqueeze(1)


# --------------------------------------------------------------------------- #
# stateless penalties (Frank's sqrt-form + customs)                           #
# --------------------------------------------------------------------------- #
def lin_vel_z_frank(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Frank._reward_lin_vel_z: sqrt(vz^2 + 1e-6) = |vz| (world frame)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sqrt(torch.square(asset.data.root_lin_vel_w.torch[:, 2]) + 1e-6)


def ang_vel_xy_frank(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Frank._reward_ang_vel_xy: sqrt(sum(w_xy^2) + 1e-6)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sqrt(torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1) + 1e-6)


def action_rate_frank(env) -> torch.Tensor:
    """Frank._reward_action_rate: sqrt(sum((prev - act)^2) + 1e-6)."""
    diff = env.action_manager.prev_action - env.action_manager.action
    return torch.sqrt(torch.sum(torch.square(diff), dim=1) + 1e-6)


def dof_vel_frank(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Frank._reward_dof_vel: sqrt(sum(dof_vel^2) + 1e-6)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sqrt(torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1) + 1e-6)


def foot_slip_frank(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, contact_thresh: float = 1.0
) -> torch.Tensor:
    """Frank._reward_foot_slip: sum over feet of sqrt(horizontal foot speed) while in contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact = _contact_z(env, sensor_cfg, contact_thresh)
    foot_speed = torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2)
    return torch.sum(torch.sqrt(foot_speed) * contact.float(), dim=1)


def stand_still_frank(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Frank._reward_stand_still: sum(|dof - default|) when NOT commanded to move."""
    asset: Articulation = env.scene[asset_cfg.name]
    stand = ~_moving_mask(env, command_name, 0.1, 0.1)
    dev = torch.sum(
        torch.abs(asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return dev * stand.float()


def joint_pos_penalty_frank(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    moving_scale: float = 0.7,
    velocity_threshold: float = 0.5,
    yaw_vel_threshold: float = 0.5,
) -> torch.Tensor:
    """Frank._reward_joint_pos_penalty on the hip joints (pass asset_cfg with hip joint_names).
    Full penalty when standing, moving_scale (0.7x) when moving."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    body_yaw_vel = torch.abs(asset.data.root_ang_vel_b.torch[:, 2])
    cmd_moving = _moving_mask(env, command_name, 0.1, 0.1)
    error = torch.linalg.norm(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )
    moving = cmd_moving | (body_vel > velocity_threshold) | (body_yaw_vel > yaw_vel_threshold)
    return torch.where(moving, moving_scale * error, error)


def zero_hip_target_dev(
    env, command_name: str, asset_cfg: SceneEntityCfg, zero_thresh: float = 0.1
) -> torch.Tensor:
    """Frank._reward_zero_hip_target_dev: penalize hip deviation from default when the command
    is (near) zero. Uses actual hip joint_pos (PD-tracked proxy for Frank's joint_pos_target)."""
    asset: Articulation = env.scene[asset_cfg.name]
    zero_cmd = ~_moving_mask(env, command_name, zero_thresh, zero_thresh)
    dev = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(dev), dim=1) * zero_cmd.float()


def base_height_band(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_height: float = 0.425,
    max_height: float = 0.442,
    margin: float = 0.035,
) -> torch.Tensor:
    """Frank._reward_base_height_band: soft band; crouching below min is 2x worse than overshoot."""
    h = _base_height(env, asset_cfg, sensor_cfg)
    low = torch.clamp(min_height - h, min=0.0)
    high = torch.clamp(h - max_height, min=0.0)
    norm_err = (low + 0.5 * high) / max(margin, 1e-6)
    return torch.square(norm_err)


def forward_min_contact_count(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_contacts: float = 2.0,
    force_thresh: float = 5.0,
    cmd_min: float = 0.3,
    lat_max: float = 0.1,
    yaw_max: float = 0.2,
) -> torch.Tensor:
    """Frank._reward_forward_min_contact_count: on a straight-forward command, penalize having
    fewer than `min_contacts` feet on the ground (prevents bounding / skating)."""
    fwd = _forward_mask(env, command_name, cmd_min, lat_max, yaw_max)
    count = _contact_z(env, sensor_cfg, force_thresh).float().sum(dim=1)
    err = torch.clamp(min_contacts - count, min=0.0)
    return err * err * fwd.float()


def forward_base_vertical_velocity(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target: float = 0.15,
    cmd_min: float = 0.3,
    lat_max: float = 0.1,
    yaw_max: float = 0.2,
) -> torch.Tensor:
    """Frank._reward_forward_base_vertical_velocity: on straight-forward, penalize base bounce
    (|vz| above target). This is the anti-'pushed-along / bobbing' term."""
    asset: Articulation = env.scene[asset_cfg.name]
    fwd = _forward_mask(env, command_name, cmd_min, lat_max, yaw_max)
    t = max(target, 1e-6)
    vz = asset.data.root_lin_vel_w.torch[:, 2]
    err = torch.clamp(torch.abs(vz) - t, min=0.0) / t
    err = torch.clamp(err, max=3.0)
    return err * err * fwd.float()


def torque_saturation(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    effort_limit: float = 50.0,
    threshold: float = 0.85,
) -> torch.Tensor:
    """Frank._reward_torque_saturation: penalize applied torque approaching the effort limit,
    weighting the worst joint (0.3) alongside the mean (0.7)."""
    asset: Articulation = env.scene[asset_cfg.name]
    sat = torch.abs(asset.data.applied_torque.torch[:, asset_cfg.joint_ids]) / max(effort_limit, 1e-6)
    denom = max(1.0 - threshold, 1e-6)
    err = torch.clamp((sat - threshold) / denom, min=0.0)
    err = torch.clamp(err, max=3.0)
    mean_err = torch.mean(err * err, dim=1)
    max_err = torch.max(err, dim=1).values ** 2
    return 0.7 * mean_err + 0.3 * max_err


# --------------------------------------------------------------------------- #
# stateful terms (need per-step memory -> ManagerTermBase with reset)         #
# --------------------------------------------------------------------------- #
class Smoothness(ManagerTermBase):
    """Frank._reward_smoothness: sqrt(sum((a - 2*a_prev + a_prev_prev)^2) + 1e-6)."""

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
        rew = torch.sqrt(torch.sum(torch.square(jerk), dim=1) + 1e-6)
        self.last_last = self.last.clone()
        self.last = act.clone()
        return rew


class TouchdownImpact(ManagerTermBase):
    """Frank._reward_touchdown_impact: at first foot contact, penalize downward foot speed
    above a threshold (soft landings). Only while commanded to move."""

    def __init__(self, cfg: RewTerm, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.command_name: str = cfg.params["command_name"]
        self.vel_threshold: float = cfg.params.get("vel_threshold", 0.30)
        nfeet = len(self.sensor_cfg.body_ids)
        self.last_contacts = torch.zeros(env.num_envs, nfeet, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.last_contacts[env_ids] = False

    def __call__(
        self, env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, vel_threshold: float = 0.30
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        moving = _moving_mask(env, command_name, 0.1, 0.1)
        contact = _contact_z(env, sensor_cfg, 1.0)
        first = contact & (~self.last_contacts)
        foot_vz = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, 2]
        down = torch.clamp(-foot_vz - vel_threshold, min=0.0)
        penalty = torch.sum(torch.square(down) * first.float(), dim=1)
        self.last_contacts = contact.clone()
        return penalty * moving.float()


class ForwardSwingPeaks(ManagerTermBase):
    """Shared state for Frank's three forward swing-peak terms. Tracks each foot's peak
    height during its swing (while on a straight-forward command) and latches the completed
    peak at touchdown, replicating Frank._update_forward_swing_peaks. The three reward
    functions below (spread / diagonal-internal / height-cap) read this latched state.

    Feet ordered by the body_names in sensor_cfg/asset_cfg; wire them as [FL, FR, RL, RR]
    so the diagonal pairing (FL+RR vs FR+RL) matches Frank."""

    def __init__(self, cfg: RewTerm, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.command_name: str = cfg.params["command_name"]
        self.force_thresh: float = cfg.params.get("force_thresh", 5.0)
        nfeet = len(self.sensor_cfg.body_ids)
        z = torch.zeros(env.num_envs, nfeet, device=env.device)
        self.current = z.clone()
        self.completed = z.clone()
        self.valid = torch.zeros(env.num_envs, nfeet, dtype=torch.bool, device=env.device)
        self.last_contacts = torch.zeros(env.num_envs, nfeet, dtype=torch.bool, device=env.device)
        self._step = -1
        self._init = False

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        fh = _feet_height(self._env, self.asset_cfg)
        self.current[env_ids] = fh[env_ids]
        self.completed[env_ids] = fh[env_ids]
        self.valid[env_ids] = False
        self.last_contacts[env_ids] = _contact_z(self._env, self.sensor_cfg, self.force_thresh)[env_ids]

    def update(self, env):
        # run the state machine at most once per sim step (three rewards call it)
        if self._step == env.common_step_counter:
            return
        self._step = env.common_step_counter
        contact = _contact_z(env, self.sensor_cfg, self.force_thresh)
        fh = _feet_height(env, self.asset_cfg)
        fwd = _forward_mask(env, self.command_name).unsqueeze(1)
        if not self._init:
            self.current = fh.clone()
            self.completed = fh.clone()
            self.last_contacts = contact.clone()
            self._init = True
            return
        swing = (~contact) & fwd
        touchdown = contact & (~self.last_contacts) & fwd
        self.current = torch.where(swing, torch.maximum(self.current, fh), self.current)
        self.completed = torch.where(touchdown, self.current, self.completed)
        self.valid = torch.where(touchdown, torch.ones_like(self.valid), self.valid)
        self.current = torch.where(touchdown, fh, self.current)
        reset = (~fwd)
        self.current = torch.where(reset, fh, self.current)
        self.completed = torch.where(reset, fh, self.completed)
        self.valid = torch.where(reset, torch.zeros_like(self.valid), self.valid)
        self.last_contacts = contact.clone()

    def __call__(self, env, mode: str, command_name: str, sensor_cfg, asset_cfg, margin: float,
                 cap_height: float = 0.06, force_thresh: float = 5.0) -> torch.Tensor:
        self.update(env)
        fwd = _forward_mask(env, command_name).float()
        valid_env = torch.all(self.valid, dim=1).float()
        peaks = self.completed
        m = max(margin, 1e-6)
        if mode == "spread":
            spread = torch.max(peaks, dim=1).values - torch.min(peaks, dim=1).values
            err = torch.clamp((spread - m) / m, min=0.0, max=3.0)
            return err * err * fwd * valid_env
        if mode == "diagonal_internal":
            diff_a = torch.abs(peaks[:, 0] - peaks[:, 3])  # FL vs RR
            diff_b = torch.abs(peaks[:, 1] - peaks[:, 2])  # FR vs RL
            err_a = torch.clamp((diff_a - m) / m, min=0.0, max=3.0)
            err_b = torch.clamp((diff_b - m) / m, min=0.0, max=3.0)
            return 0.5 * (err_a * err_a + err_b * err_b) * fwd * valid_env
        if mode == "height_cap":
            peak_max = torch.max(peaks, dim=1).values
            err = torch.clamp((peak_max - cap_height) / m, min=0.0, max=3.0)
            return err * err * fwd * valid_env
        raise ValueError(f"unknown ForwardSwingPeaks mode: {mode}")
