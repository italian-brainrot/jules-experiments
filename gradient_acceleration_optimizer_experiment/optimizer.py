import torch
from collections import defaultdict

class GradientAcceleration:
    def __init__(self, base_optimizer, beta=0.9):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")

        self.base_optimizer = base_optimizer
        self.beta = beta

        # Expose param_groups and defaults for compatibility with schedulers
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

        # Use a separate state dictionary for this wrapper
        self.ga_state = defaultdict(dict)

        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.ga_state[p]['prev_grad'] = torch.zeros_like(p.data)
                    self.ga_state[p]['acceleration'] = torch.zeros_like(p.data)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.ga_state[p]

                prev_grad = state.get('prev_grad', torch.zeros_like(grad))
                acceleration = state.get('acceleration', torch.zeros_like(grad))

                # Calculate gradient difference
                delta_grad = grad - prev_grad

                # Update acceleration (moving average of delta_grad)
                new_acceleration = self.beta * acceleration + (1 - self.beta) * delta_grad

                # Update state
                state['prev_grad'] = grad.clone()
                state['acceleration'] = new_acceleration

                # Apply acceleration to the gradient
                p.grad.data.add_(new_acceleration)

        self.base_optimizer.step()
        return loss
