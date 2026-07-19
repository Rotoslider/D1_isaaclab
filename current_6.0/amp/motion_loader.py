import json
import os

import numpy as np
import torch


class AMPLoader:
    """Loads AMP expert transitions from retargeted quadruped motion files.

    Expected frame layout:
      root_pos(3), root_rot(4), dof_pos(12), toe_pos_local(12),
      root_lin_vel(3), root_ang_vel(3), dof_vel(12), toe_vel_local(12).
    """

    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 12
    TAR_TOE_POS_LOCAL_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    TAR_TOE_VEL_LOCAL_SIZE = 12

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE
    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE
    JOINT_POSE_START_IDX = ROOT_ROT_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE
    TAR_TOE_POS_LOCAL_START_IDX = JOINT_POSE_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE
    LINEAR_VEL_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE
    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE
    JOINT_VEL_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE
    TAR_TOE_VEL_LOCAL_START_IDX = JOINT_VEL_END_IDX
    TAR_TOE_VEL_LOCAL_END_IDX = TAR_TOE_VEL_LOCAL_START_IDX + TAR_TOE_VEL_LOCAL_SIZE
    FULL_FRAME_SIZE = TAR_TOE_VEL_LOCAL_END_IDX

    AMP_OBS_DIMS = {
        "minimal": JOINT_POS_SIZE + ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + 1,
        "with_feet": JOINT_POS_SIZE + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + TAR_TOE_POS_LOCAL_SIZE + 1,
        "with_feet_vel": (
            JOINT_POS_SIZE
            + LINEAR_VEL_SIZE
            + ANGULAR_VEL_SIZE
            + JOINT_VEL_SIZE
            + TAR_TOE_POS_LOCAL_SIZE
            + TAR_TOE_VEL_LOCAL_SIZE
            + 1
        ),
    }

    def __init__(
        self,
        device,
        time_between_frames,
        preload_transitions=False,
        num_preload_transitions=100000,
        motion_files=None,
        observation_mode="minimal",
        reorder_from_pybullet_to_isaac=False,
        command_conditioned=False,
        motion_weight_overrides=None,
        command_stage_envelopes=None,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.observation_mode = observation_mode
        self.preload_transitions = preload_transitions
        # When True, feed_forward_generator also yields a per-sample command
        # (the clip's mean root vx/vy/wz) used to train a command-conditioned
        # discriminator. The clip mean (not the instantaneous frame velocity) is
        # used so the expert "command" is as smooth as the policy's commanded
        # velocity, preventing the discriminator from cheating on oscillation.
        self.command_conditioned = command_conditioned
        self.command_stage_envelopes = sorted(
            [dict(item) for item in (command_stage_envelopes or [])],
            key=lambda item: int(item.get("start_iteration", 0)),
        )
        for envelope in self.command_stage_envelopes:
            required = ("vx_min", "vx_max", "vy_abs_max", "wz_abs_max")
            missing = [key for key in required if key not in envelope]
            if missing:
                raise ValueError(f"AMP command stage envelope is missing {missing}: {envelope}")
        if self.command_stage_envelopes and int(self.command_stage_envelopes[0].get("start_iteration", 0)) > 0:
            raise ValueError("The first AMP command stage envelope must start at iteration 0")

        if observation_mode not in self.AMP_OBS_DIMS:
            raise ValueError(f"Unsupported AMP observation mode: {observation_mode}")
        if not motion_files:
            raise ValueError("AMP requires at least one motion file.")
        self.motion_weight_overrides = {
            os.path.basename(name): float(weight)
            for name, weight in (motion_weight_overrides or {}).items()
        }

        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_lens = []
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self.trajectory_mean_cmd = []

        for motion_file in motion_files:
            motion_file = str(motion_file)
            if motion_file.lower().endswith(".npz"):
                for motion in self._load_motion_npz(motion_file):
                    self._append_motion(
                        motion_file=motion_file,
                        motion_name=motion["name"],
                        motion_data=motion["frames"],
                        frame_duration=motion["frame_duration"],
                        motion_weight=motion["motion_weight"],
                        command=motion["command"],
                        reorder_from_pybullet_to_isaac=False,
                    )
                continue

            motion_json = self._load_motion_json(motion_file)
            self._append_motion(
                motion_file=motion_file,
                motion_name=os.path.splitext(os.path.basename(motion_file))[0],
                motion_data=np.asarray(motion_json["Frames"], dtype=np.float32),
                frame_duration=float(motion_json["FrameDuration"]),
                motion_weight=float(motion_json.get("MotionWeight", 1.0)),
                command=self._command_from_json(motion_json.get("Command"), motion_file),
                reorder_from_pybullet_to_isaac=reorder_from_pybullet_to_isaac,
            )

        self.trajectory_weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        self.trajectory_raw_weights = self.trajectory_weights.copy()
        trajectory_weight_sum = self.trajectory_weights.sum()
        if trajectory_weight_sum <= 0.0:
            raise ValueError("AMP motion weights must sum to a positive value.")
        self.trajectory_weights /= trajectory_weight_sum
        self.trajectory_mean_cmd = torch.stack(self.trajectory_mean_cmd, dim=0)
        self._training_iteration = 0
        self._active_trajectory_mask = np.ones(len(self.trajectories), dtype=bool)
        self._active_preload_indices = None
        self._active_preload_probabilities = None
        self._trajectory_categories = np.asarray(
            [self._command_category(command) for command in self.trajectory_mean_cmd.detach().cpu().numpy()],
            dtype=str,
        )
        self._category_target_mass = {
            category: float(self.trajectory_weights[self._trajectory_categories == category].sum())
            for category in np.unique(self._trajectory_categories)
        }
        self._print_sampling_audit("initial")

        if preload_transitions:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            self.preloaded_traj_idxs = np.asarray(traj_idxs, dtype=np.int64)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_frame_at_time_batch(
                traj_idxs, times + self.time_between_frames
            )
            if self.command_conditioned:
                self.preloaded_cmd = self.trajectory_mean_cmd[
                    torch.as_tensor(traj_idxs, dtype=torch.long, device=self.device)
                ]
            self.set_training_iteration(0)

    @staticmethod
    def _load_motion_json(motion_file):
        with open(motion_file, "r") as f:
            return json.load(f)

    @classmethod
    def _load_motion_npz(cls, motion_file):
        required_keys = (
            "format",
            "frames",
            "sequence_lengths",
            "sequence_names",
            "frame_durations",
            "commands",
            "motion_weights",
            "joint_order",
            "frame_layout",
        )
        with np.load(motion_file, allow_pickle=False) as data:
            missing = [key for key in required_keys if key not in data]
            if missing:
                raise ValueError(f"{motion_file} is missing NPZ keys: {', '.join(missing)}")

            fmt = str(np.asarray(data["format"]).item())
            if fmt != "himloco_amp_npz_v1":
                raise ValueError(f"{motion_file} has unsupported AMP NPZ format {fmt!r}")

            frames = np.asarray(data["frames"], dtype=np.float32)
            sequence_lengths = np.asarray(data["sequence_lengths"], dtype=np.int64)
            sequence_names = np.asarray(data["sequence_names"]).astype(str)
            frame_durations = np.asarray(data["frame_durations"], dtype=np.float32)
            commands = np.asarray(data["commands"], dtype=np.float32)
            motion_weights = np.asarray(data["motion_weights"], dtype=np.float32)

        if frames.ndim != 2 or frames.shape[1] != cls.FULL_FRAME_SIZE:
            raise ValueError(
                f"{motion_file} has frames shape {frames.shape}; expected (*, {cls.FULL_FRAME_SIZE})."
            )
        if not np.all(np.isfinite(frames)):
            raise ValueError(f"{motion_file} contains NaN or inf frame values.")
        if sequence_lengths.ndim != 1 or np.any(sequence_lengths < 2):
            raise ValueError(f"{motion_file} sequence_lengths must be a 1D array with lengths >= 2.")
        num_sequences = int(sequence_lengths.shape[0])
        if int(sequence_lengths.sum()) != int(frames.shape[0]):
            raise ValueError(
                f"{motion_file} sequence_lengths sum {int(sequence_lengths.sum())} "
                f"does not match frames count {frames.shape[0]}."
            )
        if sequence_names.shape[0] != num_sequences:
            raise ValueError(f"{motion_file} sequence_names length does not match sequence_lengths.")
        if frame_durations.shape != (num_sequences,):
            raise ValueError(f"{motion_file} frame_durations must have shape ({num_sequences},).")
        if commands.shape != (num_sequences, 3):
            raise ValueError(f"{motion_file} commands must have shape ({num_sequences}, 3).")
        if motion_weights.shape != (num_sequences,):
            raise ValueError(f"{motion_file} motion_weights must have shape ({num_sequences},).")
        if not np.all(np.isfinite(commands)):
            raise ValueError(f"{motion_file} commands contains NaN or inf.")
        if not np.all(np.isfinite(frame_durations)) or np.any(frame_durations <= 0.0):
            raise ValueError(f"{motion_file} frame_durations must be finite and positive.")

        offset = 0
        motions = []
        for idx, length in enumerate(sequence_lengths.astype(int)):
            next_offset = offset + length
            motions.append(
                {
                    "name": str(sequence_names[idx]),
                    "frames": frames[offset:next_offset],
                    "frame_duration": float(frame_durations[idx]),
                    "command": commands[idx],
                    "motion_weight": float(motion_weights[idx]),
                }
            )
            offset = next_offset
        return motions

    @staticmethod
    def _command_from_json(command, motion_file):
        if command is None:
            return None
        if isinstance(command, dict):
            keys = ("cmd_vx", "cmd_vy", "cmd_wz")
            try:
                return [float(command[key]) for key in keys]
            except KeyError:
                keys = ("vx", "vy", "wz")
                try:
                    return [float(command[key]) for key in keys]
                except KeyError as exc:
                    raise ValueError(f"{motion_file} Command is missing {exc.args[0]!r}") from exc
        if isinstance(command, (list, tuple)) and len(command) == 3:
            return [float(value) for value in command]
        raise ValueError(f"{motion_file} has unsupported Command metadata: {command!r}")

    def _motion_weight_with_overrides(self, motion_file, motion_name, default_weight):
        override_keys = (
            f"{os.path.basename(motion_file)}::{motion_name}",
            motion_name,
            os.path.basename(motion_file),
        )
        for key in override_keys:
            if key in self.motion_weight_overrides:
                return self.motion_weight_overrides[key]
        return float(default_weight)

    def _append_motion(
        self,
        motion_file,
        motion_name,
        motion_data,
        frame_duration,
        motion_weight,
        command,
        reorder_from_pybullet_to_isaac,
    ):
        motion_data = np.asarray(motion_data, dtype=np.float32)
        if motion_data.ndim != 2 or motion_data.shape[1] != self.FULL_FRAME_SIZE:
            raise ValueError(
                f"{motion_file}:{motion_name} has frame shape {motion_data.shape}; "
                f"expected (*, {self.FULL_FRAME_SIZE})."
            )
        if motion_data.shape[0] < 2:
            raise ValueError(f"{motion_file}:{motion_name} must contain at least 2 frames.")
        if not np.all(np.isfinite(motion_data)):
            raise ValueError(f"{motion_file}:{motion_name} contains NaN or inf frame values.")
        if reorder_from_pybullet_to_isaac:
            motion_data = self.reorder_from_pybullet_to_isaac(motion_data)

        frame_duration = float(frame_duration)
        if not np.isfinite(frame_duration) or frame_duration <= 0.0:
            raise ValueError(f"{motion_file}:{motion_name} has invalid frame_duration={frame_duration}.")

        motion_tensor = torch.tensor(motion_data, dtype=torch.float32, device=self.device)
        amp_obs = self.frame_to_amp_obs(motion_tensor)
        self.trajectories.append(amp_obs)
        self.trajectories_full.append(motion_tensor)
        self.trajectory_names.append(motion_name)
        self.trajectory_weights.append(
            self._motion_weight_with_overrides(motion_file, motion_name, motion_weight)
        )
        self.trajectory_frame_durations.append(frame_duration)
        self.trajectory_lens.append((motion_tensor.shape[0] - 1) * frame_duration)
        self.trajectory_num_frames.append(motion_tensor.shape[0])
        if command is not None:
            command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
            if command.shape != (3,):
                raise ValueError(f"{motion_file}:{motion_name} command must have shape (3,), got {command.shape}.")
            self.trajectory_mean_cmd.append(command)
            return

        # Legacy txt motions have no explicit command, so preserve the historical
        # fallback: mean executed root vx/vy/wz in the clip.
        mean_cmd = torch.tensor(
            [
                float(motion_tensor[:, self.LINEAR_VEL_START_IDX].mean()),
                float(motion_tensor[:, self.LINEAR_VEL_START_IDX + 1].mean()),
                float(motion_tensor[:, self.ANGULAR_VEL_START_IDX + 2].mean()),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.trajectory_mean_cmd.append(mean_cmd)

    @property
    def observation_dim(self):
        return self.AMP_OBS_DIMS[self.observation_mode]

    @property
    def num_motions(self):
        return len(self.trajectories)

    @staticmethod
    def _command_category(command):
        vx, vy, wz = (float(value) for value in command)
        x, y, yaw = abs(vx) > 1e-5, abs(vy) > 1e-5, abs(wz) > 1e-5
        if yaw and (x or y):
            return "mixed_yaw"
        if yaw:
            return "turn"
        if x and y:
            return "diagonal"
        if y:
            return "lateral"
        if vx > 1e-5:
            return "forward"
        if vx < -1e-5:
            return "backward"
        return "zero"

    def _print_sampling_audit(self, label):
        commands = self.trajectory_mean_cmd.detach().cpu().numpy()
        masses = {}
        ranges = {}
        for category in ("forward", "backward", "lateral", "diagonal", "turn", "mixed_yaw", "zero"):
            indices = [idx for idx, command in enumerate(commands) if self._command_category(command) == category]
            if not indices:
                continue
            masses[category] = float(self.trajectory_weights[indices].sum())
            ranges[category] = (
                float(self.trajectory_raw_weights[indices].min()),
                float(self.trajectory_raw_weights[indices].max()),
            )
        print(f"[AMP] motion sampling {label}: num_motions={self.num_motions}, category_mass={masses}, raw_weight_ranges={ranges}")

    def set_training_iteration(self, iteration):
        iteration = max(0, int(iteration))
        previous_stage = getattr(self, "_training_stage", None)
        commands = self.trajectory_mean_cmd.detach().cpu().numpy()
        if self.command_stage_envelopes:
            stage = max(
                idx
                for idx, envelope in enumerate(self.command_stage_envelopes)
                if iteration >= int(envelope.get("start_iteration", 0))
            )
            envelope = self.command_stage_envelopes[stage]
            active = (
                (commands[:, 0] >= float(envelope["vx_min"]) - 1e-6)
                & (commands[:, 0] <= float(envelope["vx_max"]) + 1e-6)
                & (np.abs(commands[:, 1]) <= float(envelope["vy_abs_max"]) + 1e-6)
                & (np.abs(commands[:, 2]) <= float(envelope["wz_abs_max"]) + 1e-6)
            )
        else:
            stage = 0 if iteration < 500 else (1 if iteration < 1500 else 2)
            active = np.ones(len(commands), dtype=bool)
            for idx, (vx, vy, wz) in enumerate(commands):
                if abs(vx) <= 1e-5 or abs(vy) <= 1e-5 or abs(wz) > 1e-5:
                    continue
                if stage == 0:
                    active[idx] = vx <= 0.70 + 1e-6 if vx >= 0.0 else abs(vx) <= 0.50 + 1e-6
                elif stage == 1:
                    active[idx] = vx <= 0.90 + 1e-6 if vx >= 0.0 else abs(vx) <= 0.60 + 1e-6
        if not np.any(active):
            raise ValueError(f"AMP stage {stage} masks every trajectory")
        self._training_iteration = iteration
        self._training_stage = stage
        self._active_trajectory_mask = active
        if hasattr(self, "preloaded_traj_idxs"):
            self._active_preload_indices = np.flatnonzero(active[self.preloaded_traj_idxs])
            if self._active_preload_indices.size == 0:
                raise ValueError(f"AMP stage {stage} masks every preloaded transition")
            trajectory_probabilities = self._active_trajectory_probabilities()
            active_traj_idxs = self.preloaded_traj_idxs[self._active_preload_indices]
            counts = np.bincount(active_traj_idxs, minlength=self.num_motions)
            missing = np.flatnonzero((trajectory_probabilities > 0.0) & (counts == 0))
            if missing.size:
                names = [self.trajectory_names[idx] for idx in missing]
                raise ValueError(f"AMP preload has no transition for active trajectories: {names}")
            probabilities = trajectory_probabilities[active_traj_idxs] / counts[active_traj_idxs]
            self._active_preload_probabilities = probabilities / probabilities.sum()
        if previous_stage != stage:
            active_names = [name for name, enabled in zip(self.trajectory_names, active) if enabled]
            masked_names = [name for name, enabled in zip(self.trajectory_names, active) if not enabled]
            envelope_text = self.command_stage_envelopes[stage] if self.command_stage_envelopes else "legacy_diagonal"
            print(f"[AMP] stage={stage} iteration={iteration} envelope={envelope_text} active={len(active_names)} masked={masked_names}")
            self._print_active_sampling_audit()

    def _active_trajectory_probabilities(self):
        probabilities = np.zeros(self.num_motions, dtype=np.float64)
        for category, target_mass in self._category_target_mass.items():
            indices = np.flatnonzero(
                self._active_trajectory_mask & (self._trajectory_categories == category)
            )
            if indices.size == 0:
                raise ValueError(f"AMP stage masks every {category!r} trajectory")
            within = self.trajectory_weights[indices]
            within = within / within.sum()
            # A stage mask can remove high-edge anchors and renormalize the
            # remaining low-speed experts below the original probability floor.
            # Re-project inside the active set so a uniformly sampled policy
            # anchor is never more than 2x as likely as its expert counterpart.
            floor = 0.5 / float(indices.size)
            projected = np.zeros(indices.size, dtype=np.float64)
            active = np.ones(indices.size, dtype=bool)
            remaining = 1.0
            while np.any(active):
                scaled = remaining * within[active] / float(within[active].sum())
                below = scaled < floor - 1e-12
                active_indices = np.flatnonzero(active)
                if not np.any(below):
                    projected[active_indices] = scaled
                    break
                fixed = active_indices[below]
                projected[fixed] = floor
                active[fixed] = False
                remaining = 1.0 - float(projected.sum())
            probabilities[indices] = float(target_mass) * projected
        probabilities /= probabilities.sum()
        return probabilities

    def _print_active_sampling_audit(self):
        probabilities = self._active_trajectory_probabilities()
        category_mass = {}
        policy_to_expert_ratio_max = {}
        for category in np.unique(self._trajectory_categories):
            indices = np.flatnonzero(
                self._active_trajectory_mask & (self._trajectory_categories == category)
            )
            if indices.size == 0:
                continue
            mass = float(probabilities[indices].sum())
            within = probabilities[indices] / mass
            category_mass[str(category)] = mass
            policy_to_expert_ratio_max[str(category)] = float(
                (1.0 / float(indices.size)) / within.min()
            )
        print(
            "[AMP] active sampling audit: "
            f"category_mass={category_mass}, "
            f"policy_to_expert_ratio_max={policy_to_expert_ratio_max}"
        )

    def weighted_traj_idx_sample_batch(self, size):
        probabilities = self._active_trajectory_probabilities()
        return np.random.choice(
            len(self.trajectories),
            size=size,
            p=probabilities,
        )

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.time_between_frames + np.finfo(np.float32).eps
        traj_lens = np.asarray(self.trajectory_lens)
        return np.maximum(traj_lens[traj_idxs] - subst, 0.0) * np.random.uniform(size=len(traj_idxs))

    def get_frame_at_time_batch(self, traj_idxs, times):
        frames = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in np.unique(traj_idxs):
            mask = traj_idxs == traj_idx
            trajectory = self.trajectories[traj_idx]
            frame_duration = self.trajectory_frame_durations[traj_idx]
            frame_idxs = times[mask] / frame_duration
            idx_low = np.floor(frame_idxs).astype(np.int64)
            idx_high = np.minimum(idx_low + 1, trajectory.shape[0] - 1)
            blend = torch.tensor(
                frame_idxs - idx_low,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            frames[mask] = (1.0 - blend) * trajectory[idx_low] + blend * trajectory[idx_high]
        return frames

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                pool = self._active_preload_indices
                if pool is None:
                    pool = np.arange(self.preloaded_s.shape[0])
                idxs = np.random.choice(
                    pool,
                    size=mini_batch_size,
                    p=self._active_preload_probabilities,
                )
                if self.command_conditioned:
                    yield (
                        self.preloaded_s[idxs],
                        self.preloaded_s_next[idxs],
                        self.preloaded_cmd[idxs],
                    )
                else:
                    yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
            elif self.command_conditioned:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                yield (
                    self.get_frame_at_time_batch(traj_idxs, times),
                    self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames),
                    self.trajectory_mean_cmd[
                        torch.as_tensor(traj_idxs, dtype=torch.long, device=self.device)
                    ],
                )
            else:
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                yield (
                    self.get_frame_at_time_batch(traj_idxs, times),
                    self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames),
                )

    def frame_to_amp_obs(self, frame):
        joint_pos = frame[:, self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]
        joint_vel = frame[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]
        ang_vel = frame[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]
        root_z = frame[:, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]
        if self.observation_mode == "minimal":
            return torch.cat((joint_pos, ang_vel, joint_vel, root_z), dim=-1)

        lin_vel = frame[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]
        toe_pos = frame[:, self.TAR_TOE_POS_LOCAL_START_IDX:self.TAR_TOE_POS_LOCAL_END_IDX]
        if self.observation_mode == "with_feet":
            return torch.cat((joint_pos, lin_vel, ang_vel, joint_vel, toe_pos, root_z), dim=-1)

        toe_vel = frame[:, self.TAR_TOE_VEL_LOCAL_START_IDX:self.TAR_TOE_VEL_LOCAL_END_IDX]
        return torch.cat((joint_pos, lin_vel, ang_vel, joint_vel, toe_pos, toe_vel, root_z), dim=-1)

    @classmethod
    def reorder_from_pybullet_to_isaac(cls, motion_data):
        """Convert leg order FR,FL,RR,RL to FL,FR,RL,RR for joint/toe blocks."""
        data = np.copy(motion_data)
        for start, width in (
            (cls.JOINT_POSE_START_IDX, 3),
            (cls.TAR_TOE_POS_LOCAL_START_IDX, 3),
            (cls.JOINT_VEL_START_IDX, 3),
            (cls.TAR_TOE_VEL_LOCAL_START_IDX, 3),
        ):
            block = motion_data[:, start:start + width * 4]
            data[:, start:start + width * 4] = np.concatenate(
                (
                    block[:, width:2 * width],
                    block[:, 0:width],
                    block[:, 3 * width:4 * width],
                    block[:, 2 * width:3 * width],
                ),
                axis=-1,
            )
        return data
