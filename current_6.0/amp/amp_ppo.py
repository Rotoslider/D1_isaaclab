# AMP algorithm for rsl-rl-lib 5.4.1: PPO + Frank's discriminator training
# (D1_HIMLoco hybrid_ppo.py:597-677, minus the HIM estimator — accepted gap,
# same as the PPO port). The discriminator/loader/normalizer/replay objects are
# built by AmpOnPolicyRunner (which owns the env handle) and attached via
# init_amp() before training starts.
#
# Faithfulness notes vs Frank (documented deviations):
# - Frank uses ONE Adam over actor_critic + disc params; rsl_rl's optimizer is
#   built before we exist, so the disc gets its OWN Adam with Frank's per-group
#   weight decay (trunk 1e-3, head 1e-1) and its lr MIRRORS the adaptive policy
#   lr each update — same lr trajectory, disjoint param sets, ~equivalent.
# - Disc update count per iteration = num_learning_epochs * num_mini_batches,
#   batch = num_envs*num_steps//num_mini_batches: identical to Frank.
from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class AmpPPO(PPO):
    def __init__(self, *args, amp_cfg: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp_cfg = dict(amp_cfg or {})
        self.discriminator = None  # set by init_amp
        self.amp_data = None
        self.amp_replay = None
        self.amp_normalizer = None
        self.disc_optimizer = None

    def init_amp(self, discriminator, amp_data, amp_replay, normalizer, amp_cfg: dict | None = None) -> None:
        if amp_cfg:
            self.amp_cfg.update(amp_cfg)
        self.discriminator = discriminator
        self.amp_data = amp_data
        self.amp_replay = amp_replay
        self.amp_normalizer = normalizer
        self.disc_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1.0e-1},
            ],
            lr=self.learning_rate,
        )

    def update(self) -> dict[str, float]:
        # mini_batch_size must be read BEFORE super().update() clears the storage
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches

        loss_dict = super().update()
        if self.discriminator is None:
            return loss_dict

        # mirror the adaptive policy lr (Frank trains the disc in the same Adam)
        for group in self.disc_optimizer.param_groups:
            group["lr"] = self.learning_rate

        grad_pen_coef = float(self.amp_cfg.get("grad_penalty_coef", 1.0))
        mean_amp_loss = mean_grad_pen = mean_policy_pred = mean_expert_pred = 0.0
        raw_states = []

        policy_gen = self.amp_replay.feed_forward_generator(num_updates, mini_batch_size)
        expert_gen = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)
        n_done = 0
        for policy_batch, expert_batch in zip(policy_gen, expert_gen):
            p_s, p_ns, p_cmd = policy_batch
            e_s, e_ns, e_cmd = expert_batch
            raw_states.append((p_s, p_ns, e_s, e_ns))

            p_s_n = self.amp_normalizer.normalize_torch(p_s, self.device)
            p_ns_n = self.amp_normalizer.normalize_torch(p_ns, self.device)
            e_s_n = self.amp_normalizer.normalize_torch(e_s, self.device)
            e_ns_n = self.amp_normalizer.normalize_torch(e_ns, self.device)

            # LSGAN targets +-1 (hybrid_ppo.py:597-636)
            policy_d = self.discriminator(self.discriminator._assemble_input(p_s_n, p_ns_n, p_cmd))
            expert_d = self.discriminator(self.discriminator._assemble_input(e_s_n, e_ns_n, e_cmd))
            expert_loss = nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.compute_grad_pen(e_s_n, e_ns_n, e_cmd, lambda_=grad_pen_coef)

            disc_loss = amp_loss + grad_pen
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            mean_amp_loss += amp_loss.item()
            mean_grad_pen += grad_pen.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            n_done += 1

        if n_done > 0:
            # update the running normalizer on this iteration's RAW states
            # (hybrid_ppo.py:671-677: after the disc loop, before next rollout)
            with torch.no_grad():
                for p_s, p_ns, e_s, e_ns in raw_states:
                    self.amp_normalizer.update(torch.cat((p_s, p_ns), dim=0).cpu().numpy())
                    self.amp_normalizer.update(torch.cat((e_s, e_ns), dim=0).cpu().numpy())
            loss_dict["amp"] = mean_amp_loss / n_done
            loss_dict["amp_grad_pen"] = mean_grad_pen / n_done
            loss_dict["amp_policy_pred"] = mean_policy_pred / n_done
            loss_dict["amp_expert_pred"] = mean_expert_pred / n_done
        return loss_dict

    def save(self) -> dict:
        saved = super().save()
        if self.discriminator is not None:
            saved["discriminator_state_dict"] = self.discriminator.state_dict()
            saved["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
            saved["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if self.discriminator is not None and "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
            self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
        return result

    def train_mode(self) -> None:
        super().train_mode()
        if self.discriminator is not None:
            self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.discriminator is not None:
            self.discriminator.eval()
