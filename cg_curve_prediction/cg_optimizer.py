import torch
from torch.optim.optimizer import Optimizer

class ConjugateGradient(Optimizer):
    def __init__(self, params):
        super(ConjugateGradient, self).__init__(params, {})

    def step(self, closure):
        # 1. Calculate loss and gradient
        loss = closure()
        grad = self._gather_flat_grad()

        # 2. Initialize state if first iteration
        if 'd' not in self.state:
            self.state['d'] = -grad.clone()
            self.state['g_old'] = grad.clone()

        d = self.state['d']
        g_old = self.state['g_old']

        # 3. Line search to find step size alpha
        x_init = self._gather_flat_params()
        directional_derivative = torch.dot(grad, d)

        def line_search_objective(alpha):
            with torch.no_grad():
                self._set_flat_params(x_init + alpha * d)
            with torch.enable_grad():
                return closure()

        alpha = self._line_search(line_search_objective, loss, directional_derivative)

        with torch.no_grad():
            self._set_flat_params(x_init + alpha * d)

        # 4. Get new gradient
        loss_new = closure()
        g_new = self._gather_flat_grad()

        # 5. Update direction d for next iteration (Polak-Ribière)
        with torch.no_grad():
            beta = torch.dot(g_new, g_new - g_old) / g_old.dot(g_old).clamp(min=1e-8)
            beta = max(0, beta.item())

            self.state['d'] = -g_new + beta * d
            self.state['g_old'] = g_new.clone()

        return loss_new

    def _line_search(self, f, f_curr, m, alpha=1.0, c=0.5, tau=0.5):
        """
        Backtracking line search satisfying Armijo condition.
        f: function to minimize (takes alpha)
        f_curr: current function value f(0)
        m: directional derivative
        """
        while f(alpha) > f_curr + c * alpha * m:
            alpha *= tau
        return alpha

    def _gather_flat_grad(self):
        views = []
        for p in self.param_groups[0]['params']:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self.param_groups[0]['params']:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _set_flat_params(self, flat_params):
        offset = 0
        for p in self.param_groups[0]['params']:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == len(flat_params)
