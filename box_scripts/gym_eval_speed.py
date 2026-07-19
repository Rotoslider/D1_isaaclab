"""Honest speed eval for the Isaac Gym HIMLoco D1 baseline — the same protocol as
d1_render6.py on the Isaac Lab side: N envs, flat ground, fixed forward command,
all-env mean/median speed over 300 policy steps (6 s).

Run (isaacgym env, from legged_gym root):
    python gym_eval_speed.py --task d1 --load_run <run> --checkpoint <N> --headless [--vx 0.8]
"""
import isaacgym  # noqa: F401  (must import before torch)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    import os as _os0
    vx = float(_os0.environ.get("D1EVAL_VX", "0.8"))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # play.py-identical overrides (the setup known to walk), + optional flat plane
    import os as _os
    plane = _os.environ.get("D1EVAL_PLANE", "0") == "1"
    env_cfg.env.num_envs = 16
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    if plane:
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    def force_cmd():
        env.commands[:, 0] = vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        if env.commands.shape[1] > 3:
            env.commands[:, 3] = 0.0

    force_cmd()
    # settle 50 steps so every env is walking from steady state, then measure 300
    for _ in range(50):
        with torch.no_grad():
            actions = policy(obs.detach())
        obs = env.step(actions.detach())[0]
        force_cmd()

    p0 = env.root_states[:, :2].clone()
    steps = 300
    resets = torch.zeros(env.num_envs, device=env.device)
    for _ in range(steps):
        with torch.no_grad():
            actions = policy(obs.detach())
        step_out = env.step(actions.detach())
        obs = step_out[0]
        resets += step_out[3].float()
        force_cmd()
    p1 = env.root_states[:, :2]
    grav_z = env.projected_gravity[:, 2]
    print(
        f"[GYMDIAG] resets/env during 6s: mean {resets.mean().item():.1f} max {resets.max().item():.0f}"
        f" | root z: mean {env.root_states[:, 2].mean().item():.2f}"
        f" | grav_z mean {grav_z.mean().item():.2f} (-1=upright)"
        f" | mean |dof_vel| {env.dof_vel.abs().mean().item():.2f} rad/s"
        f" | obs[0,:9] {[round(v, 3) for v in obs[0, :9].tolist()]}"
        f" | env.commands[0] {[round(v, 3) for v in env.commands[0, :3].tolist()]}"
        f" | commands_scale {getattr(env, 'commands_scale', 'n/a')}"
    )

    t_total = steps * env.dt
    speeds = ((p1 - p0).norm(dim=1) / t_total).sort().values
    n = speeds.numel()
    print(
        f"[GYMEVAL] all {n} envs speed m/s: mean {speeds.mean().item():.2f}"
        f"  median {speeds[n // 2].item():.2f}  min {speeds[0].item():.2f}"
        f"  max {speeds[-1].item():.2f}  (cmd {vx}, dt {env.dt}, flat plane)"
    )


if __name__ == "__main__":
    main()
