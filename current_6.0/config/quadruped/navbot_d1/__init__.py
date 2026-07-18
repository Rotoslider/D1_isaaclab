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
