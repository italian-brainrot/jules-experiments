import torch
from torch.optim.optimizer import Optimizer

class CentripetalOptimizer(Optimizer):
    def __init__(self, base_optimizer, strength=0.1):
        if strength < 0.0:
            raise ValueError(f"Invalid strength: {strength}")
        self.base_optimizer = base_optimizer
        self.strength = strength
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

    def __getstate__(self):
        # Create a state dictionary for pickling
        return {
            'base_optimizer': self.base_optimizer.__getstate__(),
            'strength': self.strength,
            'param_groups': self.param_groups,
            'defaults': self.defaults
        }

    def __setstate__(self, state):
        # Restore the state from the dictionary
        self.base_optimizer = state['base_optimizer']
        self.strength = state['strength']
        self.param_groups = state['param_groups']
        self.defaults = state['defaults']

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Calculate the mean of the parameters
                mean_p = p.data.mean()

                # Apply the centripetal force
                p.data.add_((mean_p - p.data) * self.strength)
        return loss
