# AMP env surface for the NavBot D1: ManagerBasedRLEnv + the three things Frank's
# AMP training loop needs from the env (D1_HIMLoco legged_robot.py:112-184,330-371):
#   get_amp_observations()  — 55-D "with_feet_vel" features in DATASET order
#   get_amp_transition_mask() — command settle gating (via SlewedVelocityCommand)
#   terminal AMP states     — pre-reset next-state for reset envs, in extras
# NOTE: deliberately NOT OnlyPositiveRewardEnv — Frank's AMP configs run with
# only_positive_rewards=False (d1_config.py:496).
from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

# Feature order of the himloco_amp_npz_v1 dataset (npz joint_order / toe columns).
# Isaac Lab's articulation orders joints breadth-first (grouped by type across
# legs), so indexing through these name-resolved id lists is REQUIRED — raw
# joint_pos columns would silently scramble the discriminator features.
DATASET_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DATASET_FEET_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
AMP_OBS_DIM = 55  # dof_pos 12 + lin_vel 3 + ang_vel 3 + dof_vel 12 + toe_pos 12 + toe_vel 12 + root_z 1


class AmpVelocityEnv(ManagerBasedRLEnv):
    """Velocity task env exposing Frank's `with_feet_vel` AMP observation surface."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        robot = self.scene["robot"]
        joint_ids, joint_names = robot.find_joints(DATASET_JOINT_ORDER, preserve_order=True)
        if list(joint_names) != DATASET_JOINT_ORDER:
            raise RuntimeError(f"AMP joint mapping failed: {joint_names}")
        feet_ids, feet_names = robot.find_bodies(DATASET_FEET_ORDER, preserve_order=True)
        if list(feet_names) != DATASET_FEET_ORDER:
            raise RuntimeError(f"AMP feet mapping failed: {feet_names}")
        self._amp_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self._amp_feet_ids = torch.tensor(feet_ids, dtype=torch.long, device=self.device)

    def get_amp_observations(self) -> torch.Tensor:
        """55-D AMP features, matching legged_robot.get_amp_observations `with_feet_vel`."""
        robot = self.scene["robot"]
        data = robot.data
        num_feet = len(DATASET_FEET_ORDER)

        dof_pos = data.joint_pos.torch[:, self._amp_joint_ids]
        dof_vel = data.joint_vel.torch[:, self._amp_joint_ids]
        base_lin_vel = data.root_lin_vel_b.torch
        base_ang_vel = data.root_ang_vel_b.torch
        root_pos_w = data.root_pos_w.torch
        root_quat_w = data.root_quat_w.torch
        root_lin_vel_w = data.root_lin_vel_w.torch

        # toe pos/vel in base frame; velocity with the omega x r component removed
        # (legged_robot.py:344-360 exactly)
        feet_pos_w = data.body_pos_w.torch[:, self._amp_feet_ids]
        feet_vel_w = data.body_lin_vel_w.torch[:, self._amp_feet_ids]
        feet_quat = root_quat_w[:, None, :].expand(-1, num_feet, -1).reshape(-1, 4)
        feet_pos_local = math_utils.quat_apply_inverse(
            feet_quat, (feet_pos_w - root_pos_w[:, None, :]).reshape(-1, 3)
        ).reshape(self.num_envs, num_feet, 3)
        feet_vel_local = math_utils.quat_apply_inverse(
            feet_quat, (feet_vel_w - root_lin_vel_w[:, None, :]).reshape(-1, 3)
        ).reshape(self.num_envs, num_feet, 3)
        feet_vel_local = feet_vel_local - torch.cross(
            base_ang_vel[:, None, :].expand(-1, num_feet, -1), feet_pos_local, dim=2
        )

        # root height relative to the terrain under the robot (legged_gym
        # measure_heights=True form; on the flat AMP arena this equals raw z)
        root_z = root_pos_w[:, 2:3]
        scanner = self.scene.sensors.get("height_scanner")
        if scanner is not None:
            hits_z = torch.nan_to_num(scanner.data.ray_hits_w.torch[..., 2], posinf=0.0, neginf=0.0)
            root_z = root_z - hits_z.mean(dim=-1, keepdim=True)

        return torch.cat(
            (
                dof_pos,
                base_lin_vel,
                base_ang_vel,
                dof_vel,
                feet_pos_local.reshape(self.num_envs, -1),
                feet_vel_local.reshape(self.num_envs, -1),
                root_z,
            ),
            dim=-1,
        )

    def get_amp_transition_mask(self) -> torch.Tensor:
        term = self.command_manager.get_term("base_velocity")
        if hasattr(term, "amp_transition_mask"):
            return term.amp_transition_mask
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def amp_policy_commands(self) -> torch.Tensor:
        """The 3-D command the discriminator conditions on (Frank: env.target_commands)."""
        term = self.command_manager.get_term("base_velocity")
        return term.target_vel_b if hasattr(term, "target_vel_b") else term.command

    def step(self, action: torch.Tensor):
        # stale terminal states from the previous step must not survive into this one
        self.extras.pop("amp_terminal_states", None)
        self.extras.pop("amp_terminal_env_ids", None)
        return super().step(action)

    def _reset_idx(self, env_ids):
        # capture the true pre-reset next-state for the discriminator
        # (legged_robot.py:166-169; runner patches these into the transition batch)
        if len(env_ids) > 0:
            self.extras["amp_terminal_states"] = self.get_amp_observations()[env_ids].clone()
            self.extras["amp_terminal_env_ids"] = env_ids.clone()
        super()._reset_idx(env_ids)
