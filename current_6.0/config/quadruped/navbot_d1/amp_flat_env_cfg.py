# NavBot D1 AMP task env cfg — Frank's d1_amp_canonical_cmdcond environment
# (D1_HIMLoco d1_config.py:1087-1243) on top of the validated rough cfg.
# Phase-B reward strategy (AMP_PORT_DESIGN.md): keep the validated frank reward
# set for now; align to the cmdcond reward table as a separate, diffable step.
from isaaclab.utils.configclass import configclass

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

        # the rough cfg guards this behind `__class__.__name__ == "NavBotD1RoughEnvCfg"`
        # so subclasses can tweak weights first — this subclass is done tweaking here
        self.disable_zero_weight_rewards()
