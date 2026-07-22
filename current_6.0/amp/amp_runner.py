# AMP runner for rsl-rl-lib 5.4.1: OnPolicyRunner whose rollout loop adds
# Frank's AMP machinery (D1_HIMLoco hybrid_runner.py:330-514):
#   - builds AMPLoader / AMPDiscriminator / ReplayBuffer / Normalizer and
#     attaches them to AmpPPO via init_amp()
#   - each step: amp obs pair (terminal-patched), settle-gated style-reward
#     blend r = (1-lerp)*style + lerp*task, replay insertion
# learn() is a copy of OnPolicyRunner.learn (rsl-rl-lib 5.4.1, pinned by the
# robotlab6 env) with the AMP inserts marked "# AMP:".
from __future__ import annotations

import os
import time

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan

from .amp_discriminator import AMPDiscriminator
from .amp_utils import Normalizer
from .motion_loader import AMPLoader
from .replay_buffer import ReplayBuffer


class AmpOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        amp = dict(train_cfg["amp_cfg"])
        uenv = env.unwrapped

        self._amp_env = uenv
        self.amp_task_reward_lerp = float(amp.get("task_reward_lerp", 0.75))
        # Frank's terrain-scheduled blend (d1_amp_canonical): style pressure fades
        # as terrain hardens — the flat reference clips punish climbing gaits, so a
        # fixed blend caps the terrain curriculum. Opt-in via amp_cfg; absent = the
        # fixed-lerp behavior all prior runs used.
        self._lerp_schedule = amp.get("task_reward_lerp_schedule")
        if self._lerp_schedule:
            s = self._lerp_schedule
            print(
                f"[AMP] terrain-scheduled lerp ON: {s['lerp_lo']:.2f}->{s['lerp_hi']:.2f} "
                f"over terrain levels {s['level_lo']:.1f}->{s['level_hi']:.1f}"
            )
        self.amp_data = AMPLoader(
            device=self.device,
            time_between_frames=float(uenv.step_dt),
            preload_transitions=True,
            num_preload_transitions=int(amp.get("num_preload_transitions", 2_000_000)),
            motion_files=list(amp["motion_files"]),
            observation_mode=amp.get("observation_mode", "with_feet_vel"),
            reorder_from_pybullet_to_isaac=False,
            command_conditioned=True,
            command_stage_envelopes=amp.get("command_stage_envelopes"),
        )
        obs_dim = self.amp_data.observation_dim
        env_amp_dim = uenv.get_amp_observations().shape[-1]
        if env_amp_dim != obs_dim:
            raise RuntimeError(f"AMP obs dim mismatch: env {env_amp_dim} vs dataset {obs_dim}")
        command_dim = int(amp.get("command_dim", 3))
        self.discriminator = AMPDiscriminator(
            input_dim=obs_dim * 2,
            amp_reward_coef=float(amp.get("reward_coef", 0.08)),
            hidden_layer_sizes=list(amp.get("discr_hidden_dims", [1024, 512])),
            device=self.device,
            task_reward_lerp=self.amp_task_reward_lerp,
            command_dim=command_dim,
        ).to(self.device)
        self.amp_replay = ReplayBuffer(
            obs_dim, int(amp.get("replay_buffer_size", 1_000_000)), self.device, command_dim=command_dim
        )
        self.amp_normalizer = Normalizer(obs_dim)
        self.alg.init_amp(self.discriminator, self.amp_data, self.amp_replay, self.amp_normalizer, amp_cfg=amp)

    # -- AMP helpers -------------------------------------------------------

    def _amp_terminal_patch(self, next_amp_obs: torch.Tensor) -> torch.Tensor:
        """Pre-reset next-states for envs that reset this step (hybrid_runner.py:486-487)."""
        extras = self._amp_env.extras
        if "amp_terminal_states" in extras:
            patched = next_amp_obs.clone()
            patched[extras["amp_terminal_env_ids"]] = extras["amp_terminal_states"]
            return patched
        return next_amp_obs

    # -- training loop -----------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        # AMP: state of the current transition's start
        amp_obs = self._amp_env.get_amp_observations()
        style_reward_sum = 0.0
        style_reward_count = 0

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # AMP: snapshot settle mask + discriminator command at the
                    # transition's START (resamples inside step reset the gate)
                    prev_settled = self._amp_env.get_amp_transition_mask()
                    policy_cmds = self._amp_env.amp_policy_commands.clone()

                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # AMP: next state (post-reset for done envs) + terminal-patched pair
                    next_amp_obs = self._amp_env.get_amp_observations()
                    next_amp_obs_term = self._amp_terminal_patch(next_amp_obs)
                    now_settled = self._amp_env.get_amp_transition_mask()

                    # AMP: replay stores settled transitions (hybrid_ppo.py:447-451)
                    if prev_settled.any():
                        self.amp_replay.insert(
                            amp_obs[prev_settled], next_amp_obs_term[prev_settled], policy_cmds[prev_settled]
                        )

                    # AMP: style/task blend only on settled, non-terminal transitions
                    # (hybrid_runner.py:490-514: effective = settled & next_settled & ~dones)
                    blend_mask = prev_settled & now_settled & ~dones.bool().squeeze(-1)
                    if blend_mask.any():
                        lerp_arg = None  # None -> discriminator's fixed task_reward_lerp
                        if self._lerp_schedule:
                            s = self._lerp_schedule
                            tl = self._amp_env.scene.terrain.terrain_levels
                            tl = (tl.torch if hasattr(tl, "torch") else tl).float()
                            frac = ((tl - s["level_lo"]) / max(s["level_hi"] - s["level_lo"], 1e-6)).clamp(0.0, 1.0)
                            lerp_arg = (s["lerp_lo"] + (s["lerp_hi"] - s["lerp_lo"]) * frac)[blend_mask]
                        blended, _ = self.discriminator.predict_amp_reward(
                            amp_obs[blend_mask],
                            next_amp_obs_term[blend_mask],
                            rewards[blend_mask],
                            normalizer=self.amp_normalizer,
                            command=policy_cmds[blend_mask],
                            task_reward_lerp=lerp_arg,
                        )
                        rewards = rewards.clone()
                        rewards[blend_mask] = blended
                        style_reward_sum += blended.sum().item()
                        style_reward_count += int(blend_mask.sum().item())

                    amp_obs = next_amp_obs

                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            # AMP: surface blend coverage + blended-reward level in the loss log
            if style_reward_count > 0:
                loss_dict["amp_blend_fraction"] = style_reward_count / (
                    self.cfg["num_steps_per_env"] * self.env.num_envs
                )
                loss_dict["amp_blended_reward"] = style_reward_sum / style_reward_count
            style_reward_sum = 0.0
            style_reward_count = 0

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
