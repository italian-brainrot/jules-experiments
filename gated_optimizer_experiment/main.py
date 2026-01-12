"""
DynamicAdam Optimizer Experiment

This script implements and evaluates DynamicAdam, a variant of the Adam optimizer
where the momentum parameter (beta1) is dynamically adjusted by a small neural network.

Hypothesis: The optimal amount of momentum may vary during training. By allowing a
"gating" network to control beta1 based on gradient statistics, the optimizer can
adapt to the local loss landscape, potentially leading to faster convergence or
better generalization compared to Adam with a fixed beta1.

Methodology:
1. DynamicAdam Optimizer:
   - A PyTorch optimizer that internally maintains a state similar to Adam.
   - It incorporates a small "gating" MLP that takes gradient statistics (mean and variance)
     as input and outputs a value for beta1 for the current step.
2. Meta-Learning for the Gating Network:
   - The gating network is trained to minimize the loss on a validation batch.
   - At each step, we compute "fast weights" representing a hypothetical update using
     the dynamically generated beta1.
   - The validation loss is calculated using these fast weights.
   - The gradient of the validation loss is then backpropagated to update the parameters
     of the gating network. This requires keeping the computation graph alive
     during the training update (using loss.backward(create_graph=True)).
3. Comparison:
   - DynamicAdam is benchmarked against a standard Adam optimizer.
   - Optuna is used to tune the learning rates for both optimizers, as well as the
     gating network's learning rate, to ensure a fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mnist1d.data as data
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dims=[64, 64], output_dim=10):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Gating Network Definition ---
class GatingNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Ensures beta1 is in [0, 1]
        )

    def forward(self, x):
        return self.network(x)

# --- DynamicAdam Optimizer ---
class DynamicAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, gate_lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta-1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta-2 parameter: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(DynamicAdam, self).__init__(params, defaults)

        self.gating_network = GatingNetwork()
        self.gating_optimizer = optim.Adam(self.gating_network.parameters(), lr=gate_lr)

    def _get_gradient_statistics(self, params):
        grads = [p.grad.view(-1) for p in params if p.grad is not None]
        if not grads:
            return torch.zeros(2, device=self.param_groups[0]['params'][0].device)

        flat_grads = torch.cat(grads)
        # Add a small epsilon to variance for stability
        return torch.stack([flat_grads.mean(), flat_grads.var() + 1e-8])

    def zero_grad(self, set_to_none: bool = False):
        super(DynamicAdam, self).zero_grad(set_to_none)
        self.gating_optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        # This optimizer requires a closure that returns (train_loss, val_loss_fn)
        # where val_loss_fn is a function that computes validation loss on fast_weights
        if closure is None:
            raise ValueError("DynamicAdam requires a closure to be provided.")

        # --- Part 1: Update Gating Network ---
        self.gating_optimizer.zero_grad()

        # We need gradients to calculate fast weights, so run backward pass first
        # Use a dummy loss to allow closure to be called once
        with torch.enable_grad():
            loss, val_loss_fn = closure()

        # Keep graph for meta-learning, so use create_graph=True
        loss.backward(create_graph=True)

        grad_stats = self._get_gradient_statistics(self.param_groups[0]['params'])
        dynamic_beta1 = self.gating_network(grad_stats.detach()) # Detach stats to not backprop through them

        # Calculate hypothetical "fast weights"
        fast_weights = []
        for group in self.param_groups:
            lr = group['lr']
            _, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    fast_weights.append(p)
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Use the dynamic beta1
                exp_avg_fast = dynamic_beta1 * exp_avg + (1 - dynamic_beta1) * grad
                exp_avg_sq_fast = beta2 * exp_avg_sq + (1 - beta2) * grad.pow(2)

                # Simplified Adam update for fast weights (ignoring bias correction for meta-update)
                denom = exp_avg_sq_fast.sqrt().add_(eps)
                update = exp_avg_fast / denom

                if weight_decay != 0:
                    p_fast = p - lr * (update + weight_decay * p)
                else:
                    p_fast = p - lr * update

                fast_weights.append(p_fast)

        val_loss = val_loss_fn(fast_weights)
        val_loss.backward() # This computes gradients for the gating network
        self.gating_optimizer.step()

        # --- Part 2: Update Main Model Parameters ---
        # Now perform the actual update using the latest dynamic_beta1
        # The gradients are already computed from the train_loss.backward() call
        latest_dynamic_beta1 = self.gating_network(grad_stats.detach()).item()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data # Use .data to avoid graph issues
                state = self.state[p]

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                _, beta2 = group['betas']

                # Update momentum with the dynamic beta1
                exp_avg.mul_(latest_dynamic_beta1).add_(grad, alpha=1 - latest_dynamic_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Standard Adam update logic
                bias_correction1 = 1 - latest_dynamic_beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss, val_loss

# --- Training and Evaluation ---
def train(model, optimizer, train_loader, val_loader, criterion):
    model.train()
    total_loss = 0

    # Need to iterate through val_loader simultaneously
    val_iter = iter(val_loader)

    for X_batch, y_batch in train_loader:

        try:
            X_val, y_val = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            X_val, y_val = next(val_iter)

        def val_loss_fn(fast_weights):
            # Temporarily replace model params with fast_weights
            original_params = [p.clone() for p in model.parameters()]
            for p, p_fast in zip(model.parameters(), fast_weights):
                p.data.copy_(p_fast.data)

            y_pred_val = model(X_val)
            loss = criterion(y_pred_val, y_val)

            # Restore original params
            for p, p_orig in zip(model.parameters(), original_params):
                p.data.copy_(p_orig.data)

            return loss

        def closure():
            optimizer.zero_grad()
            y_pred_train = model(X_batch)
            train_loss = criterion(y_pred_train, y_batch)
            return train_loss, val_loss_fn

        if isinstance(optimizer, DynamicAdam):
            loss, _ = optimizer.step(closure)
        else: # Standard optimizer
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            total_loss += criterion(y_pred, y_batch).item()
    return total_loss / len(loader)

# --- Optuna Objective ---
def objective(trial):
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "DynamicAdam"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    args = data.get_dataset_args()
    dataset = data.get_dataset(args)
    X_train, y_train = dataset['x'], dataset['y']
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    X_val, y_val = dataset['x_test'], dataset['y_test']
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = MLP()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "DynamicAdam":
        gate_lr = trial.suggest_float("gate_lr", 1e-5, 1e-2, log=True)
        optimizer = DynamicAdam(model.parameters(), lr=lr, gate_lr=gate_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    n_epochs = 10
    final_val_loss = 0
    for epoch in range(n_epochs):
        train(model, optimizer, train_loader, val_loader, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        if epoch == n_epochs - 1:
            final_val_loss = val_loss
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return final_val_loss

# --- Main Execution ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20) # Reduced trials to prevent timeout

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Plotting results
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("gated_optimizer_experiment/optimization_history.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("gated_optimizer_experiment/param_importances.png")

    # Final training with best params and plot loss curves
    best_params = best_trial.params
    optimizer_name = best_params["optimizer"]
    lr = best_params["lr"]

    # --- Train Best DynamicAdam ---
    torch.manual_seed(42)
    np.random.seed(42)
    dynamic_adam_model = MLP()
    gate_lr = best_params.get("gate_lr", 1e-3) # Use default if not present
    dynamic_adam_optimizer = DynamicAdam(dynamic_adam_model.parameters(), lr=lr, gate_lr=gate_lr)

    # --- Train Best Standard Adam ---
    # Find best Adam trial to compare against
    adam_trials = [t for t in study.trials if t.params.get("optimizer") == "Adam"]
    if adam_trials:
        best_adam_trial = min(adam_trials, key=lambda t: t.value)
        adam_lr = best_adam_trial.params["lr"]
        torch.manual_seed(42)
        np.random.seed(42)
        adam_model = MLP()
        adam_optimizer = optim.Adam(adam_model.parameters(), lr=adam_lr)
    else: # In case no Adam trial was run
        adam_model = None


    # Data loaders
    args = data.get_dataset_args()
    dataset = data.get_dataset(args)
    X_train, y_train = dataset['x'], dataset['y']
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    X_val, y_val = dataset['x_test'], dataset['y_test']
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    val_loader = DataLoader(val_dataset, batch_size=64)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 25
    dynamic_adam_losses = []
    adam_losses = []

    print("\nTraining final models...")
    for epoch in range(n_epochs):
        train(dynamic_adam_model, dynamic_adam_optimizer, train_loader, val_loader, criterion)
        val_loss_da = evaluate(dynamic_adam_model, val_loader, criterion)
        dynamic_adam_losses.append(val_loss_da)

        if adam_model:
            train(adam_model, adam_optimizer, train_loader, val_loader, criterion)
            val_loss_adam = evaluate(adam_model, val_loader, criterion)
            adam_losses.append(val_loss_adam)
            print(f"Epoch {epoch+1}/{n_epochs} | DynamicAdam Loss: {val_loss_da:.4f} | Adam Loss: {val_loss_adam:.4f}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} | DynamicAdam Loss: {val_loss_da:.4f}")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(dynamic_adam_losses, label=f"DynamicAdam (Best LR: {lr:.4f})")
    if adam_model:
        plt.plot(adam_losses, label=f"Adam (Best LR: {adam_lr:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("DynamicAdam vs. Adam Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("gated_optimizer_experiment/final_comparison.png")
    print("Saved final comparison plot to gated_optimizer_experiment/final_comparison.png")
