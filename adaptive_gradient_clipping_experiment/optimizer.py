import torch
from torch.optim.optimizer import Optimizer

class AdaptiveClippingOptimizer(Optimizer):
    def __init__(self, base_optimizer, alpha=1.5, beta=0.99):
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"Invalid beta value: {beta}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.beta = beta
        self.moving_average = 0.0
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Calculate global gradient norm
        global_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                global_norm += p.grad.data.norm(2).item() ** 2
        global_norm = global_norm ** 0.5

        # Update moving average of the global norm
        self.moving_average = self.beta * self.moving_average + (1 - self.beta) * global_norm

        # Set clipping threshold
        clip_threshold = self.alpha * self.moving_average

        # Clip gradients
        if clip_threshold > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for group in self.param_groups for p in group['params'] if p.grad is not None),
                clip_threshold
            )

        self.base_optimizer.step()

        return loss
