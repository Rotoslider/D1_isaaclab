# Replicates legged_gym's cfg.rewards.only_positive_rewards (used by Frank's D1RoughCfg):
# clip the per-step TOTAL reward to >= 0 so the accumulated penalties can only reduce the
# positive (tracking) reward toward zero, never drive it net-negative. Without this the
# gradient tells the policy to freeze its actions to dodge penalties instead of tracking.
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class OnlyPositiveRewardEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with legged_gym-style only_positive_rewards on the total reward."""

    def step(self, action: torch.Tensor):
        result = super().step(action)
        rew = torch.clip(result[1], min=0.0)
        self.reward_buf = rew
        return (result[0], rew) + tuple(result[2:])
