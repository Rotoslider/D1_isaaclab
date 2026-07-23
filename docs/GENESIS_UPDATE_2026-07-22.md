# D1 Project Update for Genesis — 2026-07-19 → 07-22 (the three days you were offline)

## The two machines, and how their story ended
The **Ada box** (P3 Tiny) ran its final act: Frank's own Isaac Gym stack trained the canonical AMP task
(`d1_amp_canonical_cmdcond`, 4096 envs × 10k iters) as our ground-truth baseline — **38 h 44 m at 90 °C**.
It is now an archive and MuJoCo-capable spare; no training happens there anymore. The **Blackwell nuc**
(RTX PRO 6000, driver 595 fixed its renderer) became the sole platform: the entire AMP port was built and
validated on it in one day, and it trains the same recipe **2× faster at 60 °C, half-idle**.

## RL strategy: the AMP port (the 19.5 h run and what fed it)
Frank's AMP core turned out to be pure PyTorch — we vendored it verbatim (LSGAN discriminator, targets ±1,
style reward `0.08·clamp(1−¼(d−1)²)`, blend r = 0.25·style + 0.75·task, command-conditioned input 55×2+3,
zero-centered grad penalty on expert, 1M replay, running normalizer) and wrote the Isaac Lab side: 55-dim
`with_feet_vel` AMP observations (dataset joint order!), command slew + 0.88 s settle gating, terminal-state
patching, no RSI, only-positive-rewards OFF. His 44-term cmdcond reward table was extracted and Stage-1
aligned (33 terms — including the four speed-shortfall penalties that fixed tracking). PPO replaces HIM
(accepted gap; 6-frame obs history = the deployed 270-dim interface).

## Gaits: from tap-dance to reference-matched trot
- **Interim-reward scratch run**: fast (0.97 m/s) but churning — discriminator saturated ±0.97, style dead.
- **cmdcond01 (faithful, 10k iters, 19.5 h)**: discriminator stayed engaged (±0.78) the whole run and
  sculpted the gait onto the references — **cadence 2.24 Hz vs clips' 2.25 Hz**, speeds 0.47 @ cmd 0.5 and
  0.71 @ 0.8 with tight spreads. **Head-to-head vs the Ada baseline (0.48 / 0.74): parity within 2–4 %**,
  and both stacks undershoot ~10 % identically → recipe property, not port error. Donny signed off visually.

## Terrain: two wins, one honest null, one live experiment
- **amprough01** (warm-start +3000 it): terrain curriculum to level 3.3, 0.73 m/s on rough, flat unchanged —
  gait preserved while learning footing. Donny's verdict on hard ground: falls only on extreme tiles.
- **amprough02** (+4000 it): **null result — plateau at level ~3.4 is structural.** The reference clips are
  flat-ground trots, so the discriminator punishes exactly the climbing motions hard tiles demand.
- **amprough03 (running now)**: Frank's own fix, finally ported — **terrain-scheduled blend** (style fades
  0.75→0.90 task-weight over levels 2→4, per-robot) plus a **homestead terrain mix**: 20 % discrete
  "trail rocks" (0.2–0.5 m wide, ≤0.35 m tall — Donny's real trails have 200–400 mm rocks, rare logs),
  stairs halved, roughness up. Success bar: terrain level meaningfully past 3.5.

## Deployment: the sim2sim gate is passed
Frank's replaced `rl_sar` (new policy weights only; interface unchanged) now builds on the nuc. cmdcond01
walks in **MuJoCo through the genuine deployment stack** up to the 0.90 m/s envelope cap. Getting there
found three real landmines, all fixed and documented: ONNX external-weight sidecar (now embedded,
single-file), **joint_mapping leg-pair swap** (policy orders FR,FL,RR,RL; hardware FL,FR,RL,RR — identity
mapping mirrored the controls: stable stand, violent flip on motion), and the observation layout
(term-major history, oldest-first, commands unscaled — all empirically probed). A custom 40×40 m walled
MuJoCo field with rolling relief serves as the worst-case test range. Everything — port, deploy configs,
warm-startable milestone checkpoints, videos — is public at **github.com/Rotoslider/D1_isaaclab**
(releases v1.0-cmdcond01, v1.1-amprough01).

## What's next, and where papers could help
Next: amprough03 verdict tonight → then Stage-2 reward terms (touchdown quality; needs a physics-substep
integrator), Frank's mirror-augmentation/symmetry extras, and rock-specific curriculum toward the 200–400 mm
spec. Robot arrives any week: Frank's new README is a full Jetson Orin Nano deploy manual (DaMiao CAN-FD +
DM-IMU-L1, gated suspended-first-power flow); first task is CAN motor characterization vs our DR ranges.
**Paper-hunting directions that would genuinely help:** (1) resolving the style-vs-terrain conflict — AMP
with terrain-aware or multi-skill reference data; (2) learned fall recovery / get-up policies (we fall
rarely but don't always recover); (3) perceptive locomotion — we are deliberately blind (proprioception
only); elevation-map or learned-exteroception policies are the next capability class; (4) velocity-tracking
undershoot in only-positive-free reward stacks; (5) sim2sim/sim2real gap quantification for MuJoCo-vs-PhysX
contact models.
