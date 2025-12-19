import torch
from torch.optim.optimizer import Optimizer
import math

class LRGAdam(Optimizer):
    def __init__(self, params, rank=10, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.rank = rank

        # We wrap a standard Adam optimizer
        self.base_optimizer = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Collect all gradients and flatten
        all_grads = []
        param_shapes = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                all_grads.append(p.grad.view(-1))
                param_shapes.append(p.grad.shape)

        if not all_grads:
            return loss

        flat_grads = torch.cat(all_grads)
        total_params = flat_grads.numel()

        # 2. Reshape into a matrix (padding if necessary)
        side = math.ceil(math.sqrt(total_params))
        padded_total = side * side
        padded_grads = torch.zeros(padded_total, device=flat_grads.device)
        padded_grads[:total_params] = flat_grads
        grad_matrix = padded_grads.view(side, side)

        # 3. Compute low-rank approximation via SVD
        try:
            U, S, Vh = torch.linalg.svd(grad_matrix, full_matrices=False)
            S_rank = S.clone()
            S_rank[self.rank:] = 0
            approx_grad_matrix = U @ torch.diag(S_rank) @ Vh
        except torch.linalg.LinAlgError:
            # If SVD fails, just use the original gradients
            approx_grad_matrix = grad_matrix


        # 4. Flatten back and remove padding
        approx_flat_grads = approx_grad_matrix.view(-1)[:total_params]

        # 5. Replace original gradients with the low-rank version
        current_pos = 0
        param_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                shape = param_shapes[param_idx]
                num_params = p.grad.numel()
                p.grad.copy_(approx_flat_grads[current_pos:current_pos + num_params].view(shape))
                current_pos += num_params
                param_idx += 1

        # 6. Call the internal Adam optimizer's step
        self.base_optimizer.step()

        return loss
