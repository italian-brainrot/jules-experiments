import torch
from torch.optim.optimizer import Optimizer
import numpy as np
from scipy.optimize import curve_fit

def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

class AcceleratedConjugateGradient(Optimizer):
    def __init__(self, params, n_startup_iterations=20, n_total_iterations=100):
        super(AcceleratedConjugateGradient, self).__init__(params, {})
        self.n_startup_iterations = n_startup_iterations
        self.n_total_iterations = n_total_iterations
        self.param_history = []
        self.iter = 0

    def step(self, closure):
        self.iter += 1

        if self.iter <= self.n_startup_iterations:
            # Standard CG step
            loss = self._standard_cg_step(closure)
            self.param_history.append(self._gather_flat_params().clone().detach().numpy())
            return loss
        elif self.iter == self.n_startup_iterations + 1:
            # Predict and jump
            self._predict_and_jump()
            loss = closure()
            return loss
        else:
            # Continue with standard CG from the new point
            return self._standard_cg_step(closure)

    def _standard_cg_step(self, closure):
        loss = closure()
        grad = self._gather_flat_grad()

        if 'd' not in self.state:
            self.state['d'] = -grad.clone()
            self.state['g_old'] = grad.clone()

        d = self.state['d']
        g_old = self.state['g_old']

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

        loss_new = closure()
        g_new = self._gather_flat_grad()

        with torch.no_grad():
            beta = torch.dot(g_new, g_new - g_old) / g_old.dot(g_old).clamp(min=1e-8)
            beta = max(0, beta.item())

            self.state['d'] = -g_new + beta * d
            self.state['g_old'] = g_new.clone()

        return loss_new

    def _predict_and_jump(self):
        trajectories = np.array(self.param_history)
        n_params = trajectories.shape[1]
        future_params = np.zeros(n_params)

        for i in range(n_params):
            t_data = np.arange(self.n_startup_iterations)
            p_data = trajectories[:, i]

            try:
                popt, _ = curve_fit(exp_decay, t_data, p_data, p0=(1, 1e-2, 0), maxfev=10000)
                future_params[i] = exp_decay(self.n_total_iterations, *popt)
            except RuntimeError:
                # If the fit fails, just use the last known position
                future_params[i] = p_data[-1]

        with torch.no_grad():
            self._set_flat_params(torch.tensor(future_params, dtype=torch.float32))

    def _line_search(self, f, f_curr, m, alpha=1.0, c=0.5, tau=0.5):
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
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == len(flat_params)
