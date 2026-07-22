# NavBot D1 AMP rough-terrain stage — OUR extension beyond Frank's task set
# (he has no rough cmdcond task; his rough canonical `d1_amp_canonical` is
# non-command-conditioned). Keeps the full cmdcond machinery (slewed commands,
# settle gating, conditioned discriminator, Stage-1 aligned rewards) and turns
# the terrain curriculum back on so a cmdcond01 warm start learns footing
# without losing its reference-matched gait.
import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains.height_field.hf_terrains_cfg import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils.configclass import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .amp_flat_env_cfg import NavBotD1AmpFlatEnvCfg


@configclass
class NavBotD1AmpRoughEnvCfg(NavBotD1AmpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # own copy of the generator — the module-level ROUGH_TERRAINS_CFG is shared
        # with other tasks and must not see these mutations
        self.scene.terrain.terrain_generator = copy.deepcopy(self.scene.terrain.terrain_generator)
        # undo the flat overrides: full difficulty span + terrain curriculum
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # homestead terrain mix (user's real world: trail rocks 200-400mm, rare
        # logs, few stairs): add a discrete-obstacle "trail rocks" tile and shift
        # weight from stairs toward rough/obstacles. Proportions sum to 1.0.
        sub = self.scene.terrain.terrain_generator.sub_terrains
        sub["trail_rocks"] = HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_width_range=(0.2, 0.5),
            obstacle_height_range=(0.05, 0.35),
            num_obstacles=12,
            platform_width=1.5,
            border_width=0.25,
        )
        sub["pyramid_stairs"].proportion = 0.10
        sub["pyramid_stairs_inv"].proportion = 0.10
        sub["boxes"].proportion = 0.15
        sub["random_rough"].proportion = 0.25
