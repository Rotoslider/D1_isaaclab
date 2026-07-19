# D1 AMP CmdCond → Isaac Lab reward-alignment map (agent-extracted 2026-07-19)

Source of truth: D1_HIMLoco a88b463 `D1AMPCanonicalCmdCondCfg` (d1_config.py:1245-1522,
inheritance CmdCond→Canonical→D1AMPCfg→D1RoughCfg, last assignment wins). Formulas:
d1.py `_reward_*` + legged_robot.py. Our side: navbot_d1 rough_env_cfg.py + mdp/frank_rewards.py.
44 nonzero effective terms.

**Global:** tracking_sigma=0.20 (bare denom `exp(-err/σ)` → Isaac `std=sqrt(0.20)≈0.4472`);
only_positive_rewards=False (D1AMPCfg:496 — do NOT clip); termination=-100.0 is an ordinary
negative term; dt scaling matches both frameworks → weights transcribe 1:1.

## A. Core terms (port equivalents exist — config edits)

| term | eff wt | formula | port status |
|---|---|---|---|
| tracking_lin_vel | **3.0** | exp(-Σ(cmd_xy−v_xy)²/σ) legged_robot:1419 | track_lin_vel_xy_exp wt 2.0 → 3.0, std=sqrt(0.20) |
| tracking_ang_vel | **1.8** | exp(-(cmd_yaw−ω_z)²/σ) :1424 | track_ang_vel_z_exp wt 0.5 → 1.8, std=sqrt(0.20) |
| termination | **-100.0** | reset&~timeout :1487 | is_terminated 0.0 → -100.0 |
| collision | **-5.0** | Σ(‖F‖>0.1 on non-foot) :1483 | undesired_contacts -1.0 → -5.0 |
| lin_vel_z | **-2.0** | v_z² SQUARE d1:3006 | use lin_vel_z_l2 (NOT lin_vel_z_frank sqrt-form) |
| orientation | **-0.25** | Σ proj_grav_xy² :1437 | flat_orientation_l2 -0.2 → -0.25 |
| base_height_band | **-0.25** | soft band d1:3191 | wt -0.60 → -0.25; params → min 0.360 / max 0.425 / margin 0.035 |
| feet_stumble | -0.2 | ‖F_xy‖>4·Fz, gated terrain_level≥2 d1:3034 | SKIP: dead on plane (level 0) in Frank's cmdcond too |
| has_contact | **+2.0** | standing·Σcontact/4, standing=~moving(0.1,0.05), Fz>1.0 d1:3055 | MISSING → implement |
| stand_still | **-0.20** | Σ\|q−q_def\|·~moving(0.1,0.1) d1:3074 | stand_still_frank identical, keep |
| stuck | **-1.0** | (‖v_xy‖<0.1 & \|ω_z\|<0.1 & moving_cmd) d1:3078 | MISSING → implement |
| ang_vel_xy | **-0.020** | Σω_xy² SQUARE d1:3009 | use ang_vel_xy_l2 (not frank sqrt) |
| action_rate | **-0.040** | Σ(a_prev−a)² SQUARE d1:3012 | use action_rate_l2 (not frank sqrt) |
| smoothness | **-0.006** | Σ(a−2a₁+a₂)² SQUARE d1:3015 | Smoothness class is sqrt-form → square variant, wt -0.006 |
| torque_saturation | **-0.030** | 0.7·mean(err²)+0.3·max(err)², thr 0.85 d1:3889 | wt -0.045 → -0.030 |
| touchdown_impact | **-0.010** | first-contact vdown² d1:3331 | wt -0.03 → -0.010; vel_threshold 0.30 → **0.35** |
| joint_power | **-8.0e-6** | Σ\|q̇\|·\|τ\| :1445 | wt -2e-5 → -8e-6 |
| hip_outward_weak_inward | **-0.015** | mean(((q_hip·sign−tgt)/margin)²) d1:3128 | MISSING; front -0.055/rear -0.065, margin 0.05 |
| hip_target_inward_limit | **-0.012** | mean(clamp(limit−q·sign)²/margin)·moving d1:3142 | MISSING; front -0.090/rear -0.100, margin 0.04, moving 0.1/0.2 |

## B. Zero-command hold (MISSING; zero_cmd analog = lin<0.06 & |yaw|<0.06)

zero_action_rate -0.08 (Σ(a−a_prev)²·zero d1:3113); zero_feet_vel -0.08 (Σfeet_vel²·zero :3119);
zero_dof_vel -0.015 (:3116); zero_base_vel_z -0.03 (:3122); zero_yaw_rate -0.03 (:3125).

## C. Command-shortfall (MISSING; biggest missing weights)

| term | wt | formula d1.py | params |
|---|---|---|---|
| command_xy_speed_shortfall | **-0.80** | (clamp(cmd_speed−aligned_speed−margin)/(cmd_speed+bias))²·clamp(·,err_cap), active cmd_speed≥cmd_min :3622 | cmd_min .075 margin .015 bias .10 cap 2.0 |
| command_yaw_rate_shortfall | **-0.35** | yaw version :3637 | cmd_min .10 margin .03 bias .15 cap 2.0 |
| lateral_speed_shortfall | **-0.25** | clamp(\|cmd_y\|−aligned_vy−margin)²·lateral_mask :3651 | cmd_min .075 margin .03 fwd_max .08 yaw_max .10 |
| diagonal_component_speed_shortfall | **-0.40** | per-axis vx/vy shortfall²·diag_mask :3667 | cmd_min .075 margin .03 bias .10 cap 2.0 yaw≤.10 |

## D/E. Backward shaping + directional balance — DEFERRED STAGE 2 (13 terms, Σ|wt|≈0.8)

backward_lateral_drift -0.25, backward_yaw_drift -0.15, backward_base_height_band -0.08,
backward_base_height_cap -0.020, backward_calf_clearance -0.060, backward_knee_clearance -0.025,
backward_calf_contact -0.080, moving_calf_contact -0.040, terrain_quality_calf_clearance -0.040,
backward_foot_contact_force -0.030, backward_touchdown_impact -0.030, backward_touchdown_force_rise
-0.010, backward_rear_load_takeover -0.120, directional_touchdown_impulse_balance -0.150,
directional_support_balance -0.045, directional_touchdown_action_change -0.015.
Heavy deps: physics-substep touchdown integrator (d1.py ~L315-434), calf/knee clearance probes
(d1.py:2113-2130), AMP-settled command-bucket masks + 5000→30000-step curricula. moving_calf_contact
(-0.040, F>1.0 on calves while moving) is implementable WITHOUT substep — include in Stage 1.

## Zero out in AMP cfg (active in rough, 0.0 in cmdcond)

joint_acc_l2, joint_torques_l2, joint_pos_limits, base_height_l2, dof_vel_frank, foot_slip_frank,
joint_pos_penalty_frank, zero_hip_target_dev, forward_min_contact_count,
forward_base_vertical_velocity, forward_swing_peak_spread, forward_diagonal_internal,
forward_swing_height_cap.

## Caveat: *_frank sqrt-vs-square

frank_rewards' lin_vel_z_frank / ang_vel_xy_frank / action_rate_frank / Smoothness use
sqrt(Σ()²+1e-6); this d1.py defines the D1 overrides as plain sum-of-squares. For cmdcond use the
square (l2) forms. (dof_vel_frank/foot_slip_frank divergence moot — zeroed.)
