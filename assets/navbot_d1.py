# NavBot D1 (48:1, high-torque) articulation config for Isaac Lab / robot_lab.
# Ported from D1_HIMLoco (Isaac Gym) + the deployed rl_sar config.yaml.
# Actuator params are the 48:1 values (kp 50/55/55, kd 3.2/3.5/3.5, ~50 Nm) — PROVISIONAL,
# to be reality-checked against measured motor curves after the physical robot arrives.
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

NAVBOT_D1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/navbot/d1/urdf/d1_description.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            ".*L_hip_joint": -0.05,
            ".*R_hip_joint": 0.05,
            ".*_thigh_joint": -0.75,
            ".*_calf_joint": -0.75,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=50.0,
            saturation_effort=50.0,
            velocity_limit=6.28,
            stiffness={".*_hip_joint": 50.0, ".*_thigh_joint": 55.0, ".*_calf_joint": 55.0},
            damping={".*_hip_joint": 3.2, ".*_thigh_joint": 3.5, ".*_calf_joint": 3.5},
            friction=0.0,
        ),
    },
)
"""NavBot D1 (48:1) — DC-motor actuator model."""
