import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd



class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0,
            command_dim=0):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        # input_dim is the size of the [state, next_state] pair. When
        # command_dim > 0 the discriminator becomes command-conditioned: a
        # per-transition command vector (target root vx/vy/wz) is appended to
        # the input. The reference (expert) command is the clip's own mean root
        # velocity, the policy command is the env's commanded velocity, so the
        # discriminator learns "the velocity inside the motion must match the
        # command" -> a slow crawl under a high forward command is off-manifold.
        self.input_dim = input_dim
        self.command_dim = int(command_dim)

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim + self.command_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def _assemble_input(self, state, next_state, command=None):
        parts = [state, next_state]
        if self.command_dim > 0:
            if command is None:
                raise ValueError(
                    f"AMPDiscriminator requires command_dim={self.command_dim}, but command is None")
            if command.shape[-1] != self.command_dim:
                raise ValueError(
                    f"AMP command dim mismatch: expected {self.command_dim}, got {command.shape[-1]}")
            command = command.to(device=state.device, dtype=state.dtype)
            parts.append(command)
        return torch.cat(parts, dim=-1)

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         expert_command=None,
                         lambda_=10):
        expert_data = self._assemble_input(expert_state, expert_next_state, expert_command)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None, command=None, task_reward_lerp=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(self._assemble_input(state, next_state, command)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            task_reward = task_reward.to(device=reward.device, dtype=reward.dtype).view_as(reward)
            lerp = self.task_reward_lerp if task_reward_lerp is None else task_reward_lerp
            if isinstance(lerp, torch.Tensor):
                lerp = lerp.to(device=reward.device, dtype=reward.dtype).view_as(reward)
                if torch.any(lerp > 0):
                    reward = self._lerp_reward(reward, task_reward, lerp)
            elif lerp > 0:
                reward = self._lerp_reward(reward, task_reward, lerp)
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r, task_reward_lerp=None):
        lerp = self.task_reward_lerp if task_reward_lerp is None else task_reward_lerp
        r = (1.0 - lerp) * disc_r + lerp * task_r
        return r
