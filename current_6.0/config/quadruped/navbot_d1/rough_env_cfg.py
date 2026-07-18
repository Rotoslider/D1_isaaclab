# NavBot D1 rough-terrain velocity task (Isaac Lab / robot_lab).
# Reward recipe = Frank's D1_HIMLoco D1RoughCfg (Isaac Gym / legged_gym), ported faithfully:
#   - terms with byte-identical formulas reuse robot_lab funcs at Frank's weights
#   - the rest use frank_rewards.py (his exact legged_gym formulas)
#   - commands, tracking sigma, and 5000-iter training all match his D1RoughCfg.
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from robot_lab.assets.navbot_d1 import NAVBOT_D1_CFG  # isort: skip


@configclass
class NavBotD1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------ Scene ------------------------------
        self.scene.robot = NAVBOT_D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/" + self.base_link_name

        # ------------------------------ Observations ------------------------------
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------ Actions ------------------------------
        self.actions.joint_pos.scale = {".*_hip_joint": 0.075, ".*_thigh_joint": 0.25, ".*_calf_joint": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ============== Events = Frank D1RoughCfg domain_rand (legged_gym), ported ==============
        # friction_range [0.2, 1.0]; randomize_restitution = False
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.2, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.2, 1.0)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
        # payload_mass_range [-1, 2] kg added to the base
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 2.0)
        # randomize_link_mass = False in Frank's cfg -> drop robot_lab's link-mass scaling
        self.events.randomize_rigid_body_mass_others = None
        # com_displacement_range [-0.05, 0.05] on the base (= robot_lab default range)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # randomize_kp/kd [0.9, 1.1] — robot_lab's default (0.5, 2.0) is FAR wider than Frank's
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        # randomize_joint_armature = True, joint_armature_range [0.8, 1.2] (scale)
        self.events.randomize_joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        # disturbance_range [-15, 15] N: in legged_gym this is a 1-sim-step kick every 8 policy
        # steps (avg |F| < 0.5 N). Approximated as a small persistent per-episode wrench.
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-2.0, 2.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-0.5, 0.5)
        # push_robots: push_interval_s 16, max_push_vel_xy 1.0
        self.events.randomize_push_robot.interval_range_s = (16.0, 16.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        # NOT ported (no stock Isaac Lab hook): motor_strength [0.9, 1.1], action delay (HIM-specific)

        # ================= Rewards = Frank D1RoughCfg (legged_gym), ported =================
        R = self.rewards
        cmd = "base_velocity"
        # -- byte-identical robot_lab formulas -> Frank's D1RoughCfg weights --
        R.track_lin_vel_xy_exp.weight = 2.0   # tracking_lin_vel (std=sqrt(0.25) already == his sigma 0.25)
        R.track_ang_vel_z_exp.weight = 0.5    # tracking_ang_vel
        R.flat_orientation_l2.weight = -0.2   # orientation
        R.joint_acc_l2.weight = -1e-7         # dof_acc
        R.joint_power.weight = -2e-5          # joint_power
        R.joint_torques_l2.weight = -1e-5     # torques
        R.joint_pos_limits.weight = -10.0     # dof_pos_limits
        R.undesired_contacts.weight = -1.0    # collision
        R.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        R.base_height_l2.weight = -5.0        # base_height = square(h - 0.432), same formula
        R.base_height_l2.params["target_height"] = 0.432
        R.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # -- robot_lab terms replaced by Frank's exact formulas, or absent from his set: off --
        R.is_terminated.weight = 0.0
        R.lin_vel_z_l2.weight = 0.0
        R.ang_vel_xy_l2.weight = 0.0
        R.action_rate_l2.weight = 0.0
        R.stand_still.weight = 0.0
        R.joint_pos_penalty.weight = 0.0
        R.feet_slide.weight = 0.0
        R.feet_air_time.weight = 0.0
        R.feet_gait.weight = 0.0
        R.joint_mirror.weight = 0.0
        R.upward.weight = 0.0
        R.contact_forces.weight = 0.0

        # -- Frank's custom reward functions (his exact legged_gym formulas) --
        feet_c = SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])
        feet_b = SceneEntityCfg("robot", body_names=[self.foot_link_name])
        # ordered FL,FR,RL,RR so the diagonal pairing (FL+RR vs FR+RL) matches Frank
        feet_bo = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        feet_co = SceneEntityCfg("contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        hips = SceneEntityCfg("robot", joint_names=[".*_hip_joint"])
        allj = SceneEntityCfg("robot", joint_names=[".*"])
        base = SceneEntityCfg("robot", body_names=[self.base_link_name])
        base_sensor = SceneEntityCfg("height_scanner_base")

        R.lin_vel_z_frank = RewTerm(func=mdp.lin_vel_z_frank, weight=-0.5, params={"asset_cfg": base})
        R.ang_vel_xy_frank = RewTerm(func=mdp.ang_vel_xy_frank, weight=-0.08, params={"asset_cfg": base})
        R.action_rate_frank = RewTerm(func=mdp.action_rate_frank, weight=-0.08, params={})
        R.dof_vel_frank = RewTerm(func=mdp.dof_vel_frank, weight=-0.01, params={"asset_cfg": allj})
        R.foot_slip_frank = RewTerm(
            func=mdp.foot_slip_frank, weight=-0.05, params={"sensor_cfg": feet_c, "asset_cfg": feet_b}
        )
        R.stand_still_frank = RewTerm(
            func=mdp.stand_still_frank, weight=-0.2, params={"command_name": cmd, "asset_cfg": allj}
        )
        R.joint_pos_penalty_frank = RewTerm(
            func=mdp.joint_pos_penalty_frank, weight=-0.25, params={"command_name": cmd, "asset_cfg": hips}
        )
        R.zero_hip_target_dev = RewTerm(
            func=mdp.zero_hip_target_dev, weight=-2.0, params={"command_name": cmd, "asset_cfg": hips}
        )
        R.base_height_band = RewTerm(
            func=mdp.base_height_band, weight=-0.60,
            params={"asset_cfg": base, "sensor_cfg": base_sensor,
                    "min_height": 0.425, "max_height": 0.442, "margin": 0.035},
        )
        R.forward_min_contact_count = RewTerm(
            func=mdp.forward_min_contact_count, weight=-0.16,
            params={"command_name": cmd, "sensor_cfg": feet_c, "min_contacts": 2.0, "force_thresh": 5.0},
        )
        R.forward_base_vertical_velocity = RewTerm(
            func=mdp.forward_base_vertical_velocity, weight=-0.06,
            params={"command_name": cmd, "asset_cfg": base, "target": 0.15},
        )
        R.torque_saturation = RewTerm(
            func=mdp.torque_saturation, weight=-0.045,
            params={"asset_cfg": allj, "effort_limit": 50.0, "threshold": 0.85},
        )
        R.smoothness = RewTerm(func=mdp.Smoothness, weight=-0.02, params={})
        R.touchdown_impact = RewTerm(
            func=mdp.TouchdownImpact, weight=-0.03,
            params={"command_name": cmd, "sensor_cfg": feet_co, "asset_cfg": feet_bo, "vel_threshold": 0.30},
        )
        R.forward_swing_peak_spread = RewTerm(
            func=mdp.ForwardSwingPeaks, weight=-0.04,
            params={"mode": "spread", "command_name": cmd, "sensor_cfg": feet_co, "asset_cfg": feet_bo,
                    "margin": 0.010, "force_thresh": 5.0},
        )
        R.forward_diagonal_internal = RewTerm(
            func=mdp.ForwardSwingPeaks, weight=-0.035,
            params={"mode": "diagonal_internal", "command_name": cmd, "sensor_cfg": feet_co, "asset_cfg": feet_bo,
                    "margin": 0.010, "force_thresh": 5.0},
        )
        R.forward_swing_height_cap = RewTerm(
            func=mdp.ForwardSwingPeaks, weight=-0.015,
            params={"mode": "height_cap", "command_name": cmd, "sensor_cfg": feet_co, "asset_cfg": feet_bo,
                    "margin": 0.020, "cap_height": 0.060, "force_thresh": 5.0},
        )

        if self.__class__.__name__ == "NavBotD1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------ Terminations ------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------ Curriculums ------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------ Commands = Frank D1RoughCfg (symmetric) ------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-3.14, 3.14)
        self.commands.base_velocity.rel_standing_envs = 0.1   # Frank zero_command_prob = 0.1
