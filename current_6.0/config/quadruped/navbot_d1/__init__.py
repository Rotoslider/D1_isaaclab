import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-NavBot-D1-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity.only_positive_env:OnlyPositiveRewardEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:NavBotD1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavBotD1RoughPPORunnerCfg",
    },
)

# AMP task: plain ManagerBasedRLEnv subclass, NOT OnlyPositiveRewardEnv —
# Frank's AMP configs run with only_positive_rewards=False (d1_config.py:496).
gym.register(
    id="RobotLab-Isaac-Velocity-AMP-NavBot-D1-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity.amp.amp_env:AmpVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_flat_env_cfg:NavBotD1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:NavBotD1AmpPPORunnerCfg",
    },
)

# Rough-terrain AMP stage (warm-start from cmdcond01; same experiment_name so
# --resume finds its checkpoints).
gym.register(
    id="RobotLab-Isaac-Velocity-AMP-Rough-NavBot-D1-v0",
    entry_point="robot_lab.tasks.manager_based.locomotion.velocity.amp.amp_env:AmpVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_rough_env_cfg:NavBotD1AmpRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:NavBotD1AmpPPORunnerCfg",
    },
)
