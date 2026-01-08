import torch
from torch.optim.optimizer import Optimizer

class GravitationalOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, G=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        defaults = dict(lr=lr, beta=beta, G=G)
        super(GravitationalOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("GravitationalOptimizer does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p.data)

                velocity = state['velocity']
                beta = group['beta']
                G = group['G']

                # Gravitational-inspired update
                # The gradient represents the 'force'
                force = grad

                # Deflection is proportional to the force (gradient)
                deflection = G * force

                # Update velocity (momentum)
                velocity.mul_(beta).add_(deflection)

                # Update parameters
                p.data.add_(-group['lr'], velocity)

        return loss
