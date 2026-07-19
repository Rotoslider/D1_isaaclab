# AMP → Isaac Lab port design (started 2026-07-19)

Goal: reproduce Frank's canonical AMP task **`d1_amp_canonical_cmdcond`**
(D1_HIMLoco commit a88b463, the task README §5 names as the real-robot gait)
on Isaac Lab 3.0 / Isaac Sim 6.0.1, on the manager-based `navbot_d1` stack we
already validated. The Gym-side ground truth is `amp_scratch01` currently
training on the box (~45 h) — its result is the parity bar, exactly like the
restage01/frankdr01 runs were for the PPO port.

## Framework decision (2026-07-19)

Two candidates existed in the installed stack; both were surveyed in depth:

| Path | Verdict |
|---|---|
| **skrl 2.1.0 AMP agent** (only AMP in the stack; rsl-rl-lib 5.4.1 has none) | ❌ Rejected for faithfulness: BCE-style discriminator, different style-reward form, no command conditioning, no settle gating, no per-env task/style lerp. Bending it into Frank's math = rewriting its update loop while losing rsl_rl logging/dashboard/checkpoint continuity. |
| **Vendor Frank's AMP classes onto rsl-rl-lib** | ✅ Chosen. His `AMPDiscriminator`, `AMPLoader`, `ReplayBuffer`, `Normalizer` are pure torch/numpy (zero Isaac Gym imports) — they run **unmodified** on the Blackwell (smoke test passed 2026-07-19). We write the runner/algorithm subclasses + the env AMP surface ourselves. |

Note: robot_lab6 also ships a Direct-env G1 AMP example (`tasks/direct/g1_amp/`,
skrl) — useful as reference only. Rebuilding the D1 as a Direct env would
abandon the tuned manager-based stack (frank_rewards, Frank DR events, obs
history/270-dim deploy interface, terrain machinery) — not acceptable.

## What Frank's canonical cmdcond task actually is (musts for the port)

Full reference: agent report archived in this repo's history; key facts:

- **Registered algo = HybridPPO (HIM + AMP)**, not vanilla AMPPPO. We port the
  AMP half onto plain PPO (HIM unavailable in rsl-rl-lib — same accepted gap as
  the PPO port; obs history_length=6 → 270-dim already stands in).
- **AMP obs = 55-D `with_feet_vel`**: dof_pos(12), base lin vel(3), base ang
  vel(3), dof_vel(12), toe_pos_local(12), toe_vel_local(12), root_z(1) — RAW
  (no obs scales), base-frame, toe vel with ω×r removed, root_z terrain-relative.
  Discriminator input = 55*2 + command(3) = **113**.
- **Style reward** `0.08 * clamp(1 − 0.25(d−1)², min=0)`; blend
  `r = 0.25*style + 0.75*task` (`amp_reward_coef=0.08`, `lerp=0.75`).
- **LSGAN disc loss** (targets ±1), **zero-centered grad penalty on expert
  only, λ=1.0**; disc trunk wd=1e-3, head wd=1e-1, single Adam with policy.
- **Command conditioning**: expert cmd = clip's npz target command; policy cmd
  = env target command; both appended to disc input and stored in buffers.
- **Command slew + settle gating**: applied command slews toward target
  (accel/decel per axis: x 1.6/2.4, y 1.2/1.8, yaw 2.0/3.0); AMP transitions
  only count after 44 settled steps (0.88 s at tol 0.02). Obs/task rewards see
  the SLEWED command.
- **No RSI** — resets are nominal-pose (exact_rest_reset_prob=0.75), motions
  are ONLY the discriminator's expert distribution.
- **`only_positive_rewards=False`** for AMP tasks (unlike D1RoughCfg!) — the
  AMP entry point must NOT use OnlyPositiveRewardEnv.
- **Terminal AMP states**: pre-reset next-state captured for reset envs and
  patched into the discriminator batch.
- Terrain: **plane** (cmdcond is flat); commands vx∈[−0.60,0.90],
  vy∈[−0.40,0.40], wz∈[−0.50,0.50], resample 5 s, bucketed anchor sampling.
- Dataset: `d1_amp_v4_h_uniform_cmdcond.npz` (himloco_amp_npz_v1, 92 clips,
  27324×61-D frames @50 Hz, per-clip target commands + weights). dt MUST be
  0.02 s (decimation 4 × sim dt 0.005) to match transition spacing.
- PPO hypers (cmdcond): lr 5e-4 adaptive kl 0.01, γ 0.99, λ 0.95, clip 0.2,
  entropy 0.010, max_grad_norm 0.5, steps_per_env 100, epochs 5,
  minibatches 4, replay 1e6, preload 2e6, disc hidden [1024,512], 10k iters.
- Optional shipped extras (port last): sagittal mirror augmentation of AMP
  pairs (p=0.5), actor symmetry loss 0.010 + symmetry data augmentation.

## Phases

- **A. Plumbing (DONE 2026-07-19)** — vendored `amp/` package into
  `robot_lab6 .../velocity/amp/` (masters: `current_6.0/amp/` here):
  amp_discriminator, motion_loader, replay_buffer, amp_utils(Normalizer).
  Dataset copied to `robot_lab6 source/robot_lab/data/Motions/d1/`.
  `smoke_test.py` PASSED on nuc1 (obs_dim 55, 92 clips, envelope masks 24
  high-yaw clips → 68 active, disc forward + exact lerp math, replay round-trip).
- **B. Env surface** — `AmpVelocityEnv(ManagerBasedRLEnv)` (new file, NOT
  OnlyPositiveRewardEnv): `get_amp_observations()` (Isaac Lab API: articulation
  joint pos/vel, root lin/ang vel base-frame, foot body pos/vel via body_names
  FL/FR/RL/RR_foot, ω×r removal, root_z), command slew/settle machinery on top
  of the command manager (slewed command written into the term the obs/rewards
  read), `amp_transition_valid` mask, terminal AMP state capture on reset.
  New env cfg `amp_flat_env_cfg.py` (plane, cmdcond commands/DR/rewards) +
  task id `RobotLab-Isaac-Velocity-AMP-NavBot-D1-v0`.
  Reward alignment strategy: start from the validated frankdr/frankhist reward
  set for structure, then align weights/terms to D1AMPCanonicalCmdCondCfg's
  set (it re-tunes tracking 3.0/1.8, sigma 0.20, adds directional-balance and
  touchdown terms, drops others). Two-step so a reward bug is separable from
  an AMP bug.
- **C. Training integration** — `AmpOnPolicyRunner(OnPolicyRunner)` +
  `AmpPPO(PPO)` in robot_lab6 (registered via train.py wiring like rsl_rl):
  rollout loop computes amp obs pairs, style blend on settled mask, disc
  minibatch update inside PPO epochs, normalizer update per iter, checkpoint
  save/load incl. disc + normalizer (`amp_normalizer_state_dict`).
- **D. Validation** — 5-iter smoke on nuc1 → diag_rewards + short run → full
  10k-iter run (nuc1, 4096 envs; ~20 GB VRAM, plenty free). Bar: matches or
  beats the box `amp_scratch01` Gym baseline on the all-env flat eval protocol
  (`gym_eval_speed.py` equivalent), gait quality by video.

## Gaps accepted (documented, same class as PPO port)

- HIM estimator (HybridPPO) not ported — plain PPO + 6-frame obs history.
- Bucketed command anchor sampling: approximated initially with Isaac Lab
  uniform ranges + rel_standing_envs; exact buckets portable later if the
  discriminator conditioning underperforms.
