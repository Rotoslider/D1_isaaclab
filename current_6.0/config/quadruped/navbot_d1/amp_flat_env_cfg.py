# NavBot D1 AMP task env cfg — Frank's d1_amp_canonical_cmdcond environment
# (D1_HIMLoco d1_config.py:1087-1243) on top of the validated rough cfg.
# Rewards = Stage 1 of the cmdcond alignment (CMDCOND_REWARD_MAP.md sections
# A + B + C + moving_calf_contact); Stage 2 (backward shaping + directional
# balance, substep machinery) is deferred.
import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp
from robot_lab.tasks.manager_based.locomotion.velocity.amp.commands import SlewedVelocityCommandCfg

from .rough_env_cfg import NavBotD1RoughEnvCfg


@configclass
class NavBotD1AmpFlatEnvCfg(NavBotD1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- terrain: flat arena (cmdcond is mesh_type="plane"). Minimum-difficulty
        # generator terrain, NOT a true plane — a plane breaks the height-scanner
        # ray-caster (known robot_lab6 gotcha from the --flat eval work).
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
        self.curriculum.terrain_levels = None

        # ---- commands: Frank cmdcond target envelope + slew/settle gating.
        # No command curriculum (fixed stage-0 envelope from iteration 0).
        self.curriculum.command_levels_lin_vel = None
        self.commands.base_velocity = SlewedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 5.0),
            rel_standing_envs=0.05,  # cmdcond zero bucket probability
            heading_command=False,
            ranges=SlewedVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.60, 0.90),
                lin_vel_y=(-0.40, 0.40),
                ang_vel_z=(-0.50, 0.50),
            ),
        )

        # ---- domain randomization: cmdcond values (d1_config.py:1231-1243)
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.35, 1.20)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.35, 1.20)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.75, 1.25)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.03, 0.03),
        }
        # armature 0.90-1.10 (rough cfg's randomize_joint_armature is 0.8-1.2)
        self.events.randomize_joint_armature.params["armature_distribution_params"] = (0.90, 1.10)
        # kp/kd 0.90-1.10 already set by the rough cfg.
        # cmdcond has NO persistent wrench / periodic push disturbances
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # ================= Rewards = cmdcond Stage 1 (CMDCOND_REWARD_MAP.md) =================
        R = self.rewards
        cmd = "base_velocity"

        # ---- A. core terms with existing port equivalents (weights/params/variants) ----
        # tracking_sigma = 0.20 (bare denom exp(-err/sigma) -> Isaac std = sqrt(0.20))
        R.track_lin_vel_xy_exp.weight = 3.0
        R.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.20)
        R.track_ang_vel_z_exp.weight = 1.8
        R.track_ang_vel_z_exp.params["std"] = math.sqrt(0.20)
        R.is_terminated.weight = -100.0        # termination (AmpVelocityEnv does NOT clip)
        R.undesired_contacts.weight = -5.0     # collision
        R.lin_vel_z_l2.weight = -2.0           # lin_vel_z: SQUARE form (d1:3006)
        R.flat_orientation_l2.weight = -0.25   # orientation
        R.ang_vel_xy_l2.weight = -0.020        # ang_vel_xy: SQUARE form (d1:3009)
        R.action_rate_l2.weight = -0.040       # action_rate: SQUARE form (d1:3012)
        R.joint_power.weight = -8.0e-6
        R.torque_saturation.weight = -0.030
        R.base_height_band.weight = -0.25
        R.base_height_band.params["min_height"] = 0.360
        R.base_height_band.params["max_height"] = 0.425
        R.base_height_band.params["margin"] = 0.035
        R.touchdown_impact.weight = -0.010
        R.touchdown_impact.params["vel_threshold"] = 0.35
        # stand_still_frank stays at -0.2 (identical to cmdcond stand_still -0.20)

        # frank sqrt-form variants replaced by the square (l2) forms above — never run both
        R.lin_vel_z_frank.weight = 0.0
        R.ang_vel_xy_frank.weight = 0.0
        R.action_rate_frank.weight = 0.0
        # smoothness: reassign to the square form (d1:3015), wt -0.006 (sqrt Smoothness dropped)
        R.smoothness = RewTerm(func=mdp.SmoothnessSq, weight=-0.006, params={})

        # ---- entity cfgs for the new Stage-1 terms ----
        feet_c = SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])
        feet_b = SceneEntityCfg("robot", body_names=[self.foot_link_name])
        calves_c = SceneEntityCfg("contact_forces", body_names=[".*_calf"])
        allj = SceneEntityCfg("robot", joint_names=[".*"])
        # ordered FL,FR,RL,RR so the HAA outward signs [+1,-1,+1,-1] and the
        # [front,front,rear,rear] targets line up (d1.py _get_haa_outward_signs)
        hips_o = SceneEntityCfg(
            "robot",
            joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
            preserve_order=True,
        )

        # ---- A. missing core terms ----
        R.has_contact = RewTerm(
            func=mdp.has_contact, weight=2.0,
            params={"command_name": cmd, "sensor_cfg": feet_c, "force_thresh": 1.0},
        )
        R.stuck = RewTerm(func=mdp.stuck, weight=-1.0, params={"command_name": cmd})
        R.hip_outward_weak_inward = RewTerm(
            func=mdp.hip_outward_weak_inward, weight=-0.015,
            params={"asset_cfg": hips_o, "target_front": -0.055, "target_rear": -0.065, "margin": 0.05},
        )
        R.hip_target_inward_limit = RewTerm(
            func=mdp.hip_target_inward_limit, weight=-0.012,
            params={"command_name": cmd, "asset_cfg": hips_o,
                    "limit_front": -0.090, "limit_rear": -0.100, "margin": 0.04},
        )

        # ---- B. zero-command hold (zero_cmd analog mask: |lin|<0.06 & |yaw|<0.06) ----
        R.zero_action_rate = RewTerm(func=mdp.zero_action_rate, weight=-0.08, params={"command_name": cmd})
        R.zero_feet_vel = RewTerm(
            func=mdp.zero_feet_vel, weight=-0.08, params={"command_name": cmd, "asset_cfg": feet_b}
        )
        R.zero_dof_vel = RewTerm(
            func=mdp.zero_dof_vel, weight=-0.015, params={"command_name": cmd, "asset_cfg": allj}
        )
        R.zero_base_vel_z = RewTerm(func=mdp.zero_base_vel_z, weight=-0.03, params={"command_name": cmd})
        R.zero_yaw_rate = RewTerm(func=mdp.zero_yaw_rate, weight=-0.03, params={"command_name": cmd})

        # ---- C. command-shortfall ----
        R.command_xy_speed_shortfall = RewTerm(
            func=mdp.command_xy_speed_shortfall, weight=-0.80,
            params={"command_name": cmd, "cmd_min": 0.075, "margin": 0.015, "denom_bias": 0.10, "err_cap": 2.0},
        )
        R.command_yaw_rate_shortfall = RewTerm(
            func=mdp.command_yaw_rate_shortfall, weight=-0.35,
            params={"command_name": cmd, "cmd_min": 0.10, "margin": 0.03, "denom_bias": 0.15, "err_cap": 2.0},
        )
        R.lateral_speed_shortfall = RewTerm(
            func=mdp.lateral_speed_shortfall, weight=-0.25,
            params={"command_name": cmd, "cmd_min": 0.075, "margin": 0.03, "forward_max": 0.08, "yaw_max": 0.10},
        )
        R.diagonal_component_speed_shortfall = RewTerm(
            func=mdp.diagonal_component_speed_shortfall, weight=-0.40,
            params={"command_name": cmd, "cmd_min": 0.075, "margin": 0.03,
                    "denom_bias": 0.10, "err_cap": 2.0, "yaw_max": 0.10},
        )

        # ---- D. Stage-1 slice: moving_calf_contact (rest of D/E deferred to Stage 2) ----
        R.moving_calf_contact = RewTerm(
            func=mdp.moving_calf_contact, weight=-0.040,
            params={"command_name": cmd, "sensor_cfg": calves_c, "force_threshold": 1.0},
        )

        # ---- zero-out: active in the rough port, 0.0 in cmdcond ----
        R.joint_acc_l2.weight = 0.0
        R.joint_torques_l2.weight = 0.0
        R.joint_pos_limits.weight = 0.0
        R.base_height_l2.weight = 0.0
        R.dof_vel_frank.weight = 0.0
        R.foot_slip_frank.weight = 0.0
        R.joint_pos_penalty_frank.weight = 0.0
        R.zero_hip_target_dev.weight = 0.0
        R.forward_min_contact_count.weight = 0.0
        R.forward_base_vertical_velocity.weight = 0.0
        R.forward_swing_peak_spread.weight = 0.0
        R.forward_diagonal_internal.weight = 0.0
        R.forward_swing_height_cap.weight = 0.0
        # feet_stumble: SKIP — dead on plane (terrain level 0) in Frank's cmdcond too

        # the rough cfg guards this behind `__class__.__name__ == "NavBotD1RoughEnvCfg"`
        # so subclasses can tweak weights first — this subclass is done tweaking here
        self.disable_zero_weight_rewards()
