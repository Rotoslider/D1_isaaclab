# AMP runner cfg — Frank D1AMPCanonicalCmdCondCfgPPO hyperparameters
# (d1_config.py:1524-1594). algorithm.class_name uses the module:Class form
# resolve_callable accepts; the runner class needs its own branch in train.py.
from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

_D1_AMP_DATASET = f"{ISAACLAB_ASSETS_DATA_DIR}/Motions/d1/d1_amp_v4_h_uniform_cmdcond.npz"

# shared base (module-level: @configclass consumes class attributes, so the
# subclass cannot read the parent's amp_cfg after decoration)
_BASE_AMP_CFG = {
    "motion_files": [_D1_AMP_DATASET],
    "observation_mode": "with_feet_vel",
    "command_dim": 3,
    "reward_coef": 0.08,             # amp_reward_coef
    "task_reward_lerp": 0.75,        # r = 0.25*style + 0.75*task
    "grad_penalty_coef": 1.0,        # amp_grad_penalty_coef (canonical, not the 10.0 base)
    "discr_hidden_dims": [1024, 512],
    "replay_buffer_size": 1_000_000,
    "num_preload_transitions": 2_000_000,
    "command_stage_envelopes": [
        {"start_iteration": 0, "vx_min": -0.60, "vx_max": 0.90, "vy_abs_max": 0.40, "wz_abs_max": 0.50}
    ],
}


@configclass
class NavBotD1AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpOnPolicyRunner"
    num_steps_per_env = 100          # Frank D1AMPCfgPPO.runner
    max_iterations = 10000           # cmdcond.runner
    save_interval = 100              # Frank uses 20; 100 keeps the log dir sane at 10k iters
    experiment_name = "navbot_d1_amp"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="robot_lab.tasks.manager_based.locomotion.velocity.amp.amp_ppo:AmpPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.010,          # cmdcond.algorithm
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,        # cmdcond.algorithm
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,           # cmdcond.algorithm
    )
    amp_cfg = dict(_BASE_AMP_CFG)


@configclass
class NavBotD1AmpRoughPPORunnerCfg(NavBotD1AmpPPORunnerCfg):
    """Rough-terrain AMP stage: adds Frank's terrain-scheduled style/task blend
    (d1_amp_canonical mechanism, adapted: his 0.6->0.8 over levels 2->3; our base
    lerp is 0.75 so we fade 0.75->0.90 over levels 2->4). Fixes the level-3.4
    curriculum plateau — flat reference clips punish climbing gaits, so style
    pressure must fade as terrain hardens."""

    amp_cfg = {
        **_BASE_AMP_CFG,
        "task_reward_lerp_schedule": {"level_lo": 2.0, "level_hi": 4.0, "lerp_lo": 0.75, "lerp_hi": 0.90},
    }
