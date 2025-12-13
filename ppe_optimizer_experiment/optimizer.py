import torch
from torch.optim.optimizer import Optimizer
from collections import deque

class PPE(Optimizer):
    """
    Polynomial Parameter Extrapolation (PPE) Optimizer.

    This optimizer wraps a base optimizer and uses polynomial extrapolation on the
    parameter history to potentially accelerate convergence.
    """
    def __init__(self, params, base_optimizer, history_size=10, degree=2, alpha=0.5):
        """
        Initializes the PPE optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            base_optimizer: The base optimizer (e.g., Adam, SGD).
            history_size (int): The number of past parameter values to use for extrapolation.
            degree (int): The degree of the polynomial to fit.
            alpha (float): The interpolation factor between the base optimizer's update
                           and the extrapolated value. 0.0 means only the base optimizer's
                           update is used, 1.0 means only the extrapolation is used.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not history_size >= degree + 1:
            raise ValueError("History size must be at least degree + 1")

        defaults = dict(history_size=history_size, degree=degree, alpha=alpha)
        super(PPE, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        for group in self.param_groups:
            # Override default group values with the ones from base_optimizer if they exist.
            group.setdefault('history_size', history_size)
            group.setdefault('degree', degree)
            group.setdefault('alpha', alpha)
            for p in group['params']:
                self.state[p]['history'] = deque(maxlen=group['history_size'])

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                                          and returns the loss.
        """
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            degree = group['degree']
            alpha = group['alpha']
            history_size = group['history_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                history = state['history']

                history.append(p.data.clone())

                if len(history) == history_size:
                    # 1. Prepare data for polynomial fitting
                    y = torch.stack(list(history))
                    y_flat = y.view(history_size, -1)

                    x = torch.arange(history_size, device=y.device, dtype=y.dtype)

                    # 2. Fit polynomial coefficients using least squares
                    V = torch.vander(x, N=degree + 1)
                    try:
                        coeffs = torch.linalg.lstsq(V, y_flat).solution
                    except torch.linalg.LinAlgError:
                        # lstsq can fail if the history is not well-conditioned.
                        # In this case, we just skip the extrapolation for this step.
                        continue


                    # 3. Extrapolate to the next time step
                    x_next = torch.tensor(history_size, device=y.device, dtype=y.dtype)
                    x_next_powers = x_next.pow(torch.arange(degree, -1, -1, device=y.device, dtype=y.dtype))

                    extrapolated_flat = x_next_powers @ coeffs
                    extrapolated = extrapolated_flat.view_as(p.data)

                    # 4. Apply the update as a weighted average
                    p.data.add_(extrapolated - p.data, alpha=alpha)

        return loss
