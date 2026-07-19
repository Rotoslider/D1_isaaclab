"""Deployment-matched command transition helpers with no Isaac Gym dependency."""

import torch


def slew_toward_tensor(current, target, accel_rate, decel_rate, dt):
    """Vectorized equivalent of RL_SAR command_shaping.hpp::SlewToward."""
    delta = target - current
    same_sign = current * target > 0.0
    crossing_zero = current * target < 0.0
    reducing_magnitude = same_sign & (torch.abs(target) < torch.abs(current))
    moving_to_zero = torch.abs(target) <= 1.0e-6
    rate = torch.where(crossing_zero | reducing_magnitude | moving_to_zero, decel_rate, accel_rate)
    max_delta = torch.clamp(rate * float(dt), min=0.0)
    limited_delta = torch.maximum(-max_delta, torch.minimum(delta, max_delta))
    return current + limited_delta


def project_d1_stage0_command_envelope(commands):
    """Torch equivalent of RL_SAR's final Stage-0 ClampCommands projection."""
    projected = commands.clone()
    projected[:, 0] = torch.clamp(projected[:, 0], -0.60, 0.90)
    projected[:, 1] = torch.clamp(projected[:, 1], -0.40, 0.40)
    projected[:, 2] = torch.clamp(projected[:, 2], -0.50, 0.50)

    has_x = torch.abs(projected[:, 0]) > 0.05
    has_y = torch.abs(projected[:, 1]) > 0.05
    diagonal = has_x & has_y
    forward_diagonal = diagonal & (projected[:, 0] >= 0.0)
    backward_diagonal = diagonal & (projected[:, 0] < 0.0)
    projected[:, 0] = torch.where(
        forward_diagonal,
        torch.clamp(projected[:, 0], 0.0, 0.70),
        projected[:, 0],
    )
    projected[:, 1] = torch.where(
        forward_diagonal,
        torch.clamp(projected[:, 1], -0.35, 0.35),
        projected[:, 1],
    )
    projected[:, 0] = torch.where(
        backward_diagonal,
        torch.clamp(projected[:, 0], -0.50, 0.0),
        projected[:, 0],
    )
    projected[:, 1] = torch.where(
        backward_diagonal,
        torch.clamp(projected[:, 1], -0.25, 0.25),
        projected[:, 1],
    )

    has_x = torch.abs(projected[:, 0]) > 0.05
    has_y = torch.abs(projected[:, 1]) > 0.05
    has_yaw = torch.abs(projected[:, 2]) > 0.02
    translated_yaw = has_yaw & (has_x | has_y)
    forward_yaw = translated_yaw & ~has_y & (projected[:, 0] > 0.05) & (projected[:, 0] <= 0.70)
    backward_yaw = (
        translated_yaw
        & ~has_y
        & (projected[:, 0] >= -0.60)
        & (projected[:, 0] <= -0.50)
    )
    yaw_limit = torch.zeros_like(projected[:, 2])
    yaw_limit = torch.where(forward_yaw, torch.full_like(yaw_limit, 0.50), yaw_limit)
    yaw_limit = torch.where(backward_yaw, torch.full_like(yaw_limit, 0.40), yaw_limit)
    projected[:, 2] = torch.where(
        translated_yaw,
        torch.maximum(-yaw_limit, torch.minimum(projected[:, 2], yaw_limit)),
        projected[:, 2],
    )
    return projected
