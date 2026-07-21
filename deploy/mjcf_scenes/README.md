# MJCF scenes for rl_sar sim2sim

- scene_mild.xml -> goes in d1_description/mjcf/
- terrain_mild.png -> MUST go in d1_description/meshes/ (MuJoCo resolves hfield files via the model meshdir)
- run: D1_RL_CONFIG=lab_amprough01 ./cmake_build/bin/rl_sim_mujoco d1 scene_mild
