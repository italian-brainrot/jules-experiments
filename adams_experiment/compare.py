import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset_args, make_dataset
import os
import math
import optuna
import plotly.graph_objects as go

# Set a fixed random seed for reproducibility
torch.manual_seed(42)

# Load the mnist1d dataset
args = get_dataset_args()
data = make_dataset(args)

X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])

# Create data loaders
train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_train.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

class AdamS(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, sign_beta=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= sign_beta < 1.0:
            raise ValueError("Invalid sign_beta parameter: {}".format(sign_beta))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, sign_beta=sign_beta)
        super(AdamS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamS does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient signs
                    state['exp_avg_sign'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq, exp_avg_sign = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_sign']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                sign_beta = group['sign_beta']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction_sign = 1 - sign_beta ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Update the moving average of the gradient signs
                exp_avg_sign.mul_(sign_beta).add_(grad.sign(), alpha=1 - sign_beta)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # Modulate the learning rate by the absolute value of the sign stability
                sign_stability = exp_avg_sign / bias_correction_sign

                # We need to create a tensor of the same size as p, filled with the adapted learning rate
                # and then use that for the update.
                # The operation should be p.addcdiv_(exp_avg, denom, value=-1.0 * adapted_lr)
                # but adapted_lr is a tensor, and value must be a scalar.
                # So we do element-wise multiplication of adapted_lr with the update term.
                update = exp_avg / denom
                p.add_(update * -step_size * sign_stability.abs())

        return loss

def train(model, optimizer, criterion, train_loader):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def validate(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.data[0])
    accuracy = 100. * correct / len(test_loader.data[0])
    return test_loss, accuracy

def objective(trial, optimizer_name):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = MLP()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamS':
        sign_beta = trial.suggest_float('sign_beta', 0.8, 0.99)
        optimizer = AdamS(model.parameters(), lr=lr, sign_beta=sign_beta)

    val_losses = []
    for epoch in range(20):
        train(model, optimizer, criterion, train_loader)
        val_loss, _ = validate(model, criterion, test_loader)
        val_losses.append(val_loss)

    return min(val_losses)

if __name__ == '__main__':
    # --- Optuna Study ---
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=30)

    study_adams = optuna.create_study(direction='minimize')
    study_adams.optimize(lambda trial: objective(trial, 'AdamS'), n_trials=30)

    print("Best trial for Adam:", study_adam.best_trial.params)
    print("Best trial for AdamS:", study_adams.best_trial.params)

    # --- Final Comparison ---
    adam_histories = []
    adams_histories = []

    for i in range(5): # Run 5 times with different seeds for robustness
        torch.manual_seed(42 + i)

        # Train Adam with best params
        model_adam = MLP()
        optimizer_adam = optim.Adam(model_adam.parameters(), **study_adam.best_trial.params)
        criterion = nn.CrossEntropyLoss()
        history_adam = []
        for epoch in range(50):
            train(model_adam, optimizer_adam, criterion, train_loader)
            val_loss, _ = validate(model_adam, criterion, test_loader)
            history_adam.append(val_loss)
        adam_histories.append(history_adam)

        # Train AdamS with best params
        model_adams = MLP()
        optimizer_adams = AdamS(model_adams.parameters(), **study_adams.best_trial.params)
        criterion = nn.CrossEntropyLoss()
        history_adams = []
        for epoch in range(50):
            train(model_adams, optimizer_adams, criterion, train_loader)
            val_loss, _ = validate(model_adams, criterion, test_loader)
            history_adams.append(val_loss)
        adams_histories.append(history_adams)

    # Calculate mean and std dev for plotting
    adam_mean = torch.tensor(adam_histories).mean(axis=0)
    adam_std = torch.tensor(adam_histories).std(axis=0)
    adams_mean = torch.tensor(adams_histories).mean(axis=0)
    adams_std = torch.tensor(adams_histories).std(axis=0)

    # --- Plotting ---
    fig = go.Figure()

    # Adam plot
    fig.add_trace(go.Scatter(x=torch.arange(len(adam_mean)), y=adam_mean, mode='lines', name='Adam'))
    fig.add_trace(go.Scatter(
        x=torch.arange(len(adam_mean)),
        y=adam_mean + adam_std,
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=torch.arange(len(adam_mean)),
        y=adam_mean - adam_std,
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    # AdamS plot
    fig.add_trace(go.Scatter(x=torch.arange(len(adams_mean)), y=adams_mean, mode='lines', name='AdamS'))
    fig.add_trace(go.Scatter(
        x=torch.arange(len(adams_mean)),
        y=adams_mean + adams_std,
        fill='tonexty',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=torch.arange(len(adams_mean)),
        y=adams_mean - adams_std,
        fill='tonexty',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title='Adam vs. AdamS Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Validation Loss'
    )

    fig.write_image("comparison.png")
