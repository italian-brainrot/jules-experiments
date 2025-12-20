
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import numpy as np
import matplotlib.pyplot as plt
import optuna
import copy

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.layers(x)

# Data loading
defaults = get_dataset_args()
defaults.num_samples = 4000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_val, y_val = torch.tensor(data['x_test'], dtype=torch.float32), torch.tensor(data['y_test'])

train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
val_loader = TensorDataLoader((X_val, y_val), batch_size=128)

def train_meta(model, train_loader, val_loader, epochs=50):
    log_lr = nn.Parameter(torch.tensor(np.log(1e-3)))
    meta_optimizer = optim.Adam([log_lr], lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    lr_history = []

    for epoch in range(epochs):
        model.train()
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        for i in range(len(train_loader)):
            try:
                X_batch, y_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X_batch, y_batch = next(train_iter)

            try:
                X_val_batch, y_val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                X_val_batch, y_val_batch = next(val_iter)

            # Meta-update for LR
            outputs = model(X_batch)
            train_loss = criterion(outputs, y_batch)
            grad_params = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)

            fast_weights = [p - torch.exp(log_lr) * g for p, g in zip(model.parameters(), grad_params)]

            val_out_l1 = torch.nn.functional.linear(X_val_batch, fast_weights[0], fast_weights[1])
            val_out_relu = torch.nn.functional.relu(val_out_l1)
            val_out_l2 = torch.nn.functional.linear(val_out_relu, fast_weights[2], fast_weights[3])
            val_loss = criterion(val_out_l2, y_val_batch)

            meta_optimizer.zero_grad()
            val_loss.backward()
            meta_optimizer.step()

            # Actual model parameter update
            with torch.no_grad():
                for p, g in zip(model.parameters(), grad_params):
                    p.data -= torch.exp(log_lr) * g
            model.zero_grad()

        # Evaluate at the end of the epoch
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss_total += criterion(outputs, y_batch).item()

        avg_val_loss = val_loss_total / len(val_loader)
        history.append(avg_val_loss)
        current_lr = torch.exp(log_lr).item()
        lr_history.append(current_lr)
        print(f'Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')

    return history, lr_history

# Baseline Adam training
def train_adam(model, train_loader, val_loader, lr, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss_total += criterion(outputs, y_batch).item()

        avg_val_loss = val_loss_total / len(val_loader)
        history.append(avg_val_loss)
    return history

# Adam with Cosine Annealing
def train_adam_cosine(model, train_loader, val_loader, lr, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss_total += criterion(outputs, y_batch).item()

        avg_val_loss = val_loss_total / len(val_loader)
        history.append(avg_val_loss)
    return history


# Optuna for baseline Adam
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model = MLP()
    history = train_adam(copy.deepcopy(model), train_loader, val_loader, lr, epochs=10)
    return min(history)

# Run experiments
if __name__ == '__main__':
    # Meta-optimizer
    model_meta = MLP()
    history_meta, lr_history = train_meta(model_meta, train_loader, val_loader)

    # Tuned Adam
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_lr = study.best_trial.params['lr']
    print(f"Best LR for Adam: {best_lr}")
    model_adam = MLP()
    history_adam = train_adam(model_adam, train_loader, val_loader, best_lr)

    # Adam with Cosine Annealing
    model_cosine = MLP()
    history_cosine = train_adam_cosine(model_cosine, train_loader, val_loader, best_lr)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history_meta, label='Meta-Optimizer')
    plt.plot(history_adam, label=f'Tuned Adam (lr={best_lr:.4f})')
    plt.plot(history_cosine, label='Adam with Cosine Annealing')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_adaptation_experiment/comparison.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(lr_history, label='Meta-Optimizer LR')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_adaptation_experiment/lr_evolution.png')
    plt.show()
