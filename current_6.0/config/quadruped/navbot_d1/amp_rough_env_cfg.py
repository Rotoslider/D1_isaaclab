# NavBot D1 AMP rough-terrain stage — OUR extension beyond Frank's task set
# (he has no rough cmdcond task; his rough canonical `d1_amp_canonical` is
# non-command-conditioned). Keeps the full cmdcond machinery (slewed commands,
# settle gating, conditioned discriminator, Stage-1 aligned rewards) and turns
# the terrain curriculum back on so a cmdcond01 warm start learns footing
# without losing its reference-matched gait.
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from robot_lab.tasks.manager_based.locomotion.velocity import mdp

from .amp_flat_env_cfg import NavBotD1AmpFlatEnvCfg


@configclass
class NavBotD1AmpRoughEnvCfg(NavBotD1AmpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # undo the flat overrides: full difficulty span + terrain curriculum
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
