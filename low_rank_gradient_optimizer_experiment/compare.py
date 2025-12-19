import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
import copy

from .optimizer import LRGAdam

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Data Loading ---
def get_data():
    args = get_dataset_args()
    args.num_samples = 2000
    args.num_test_samples = 1000
    # Use get_dataset which handles downloading and caching
    data = get_dataset(args, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist1d_data.pkl'), download=True)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    # Use a subset for faster tuning
    X_tune, y_tune = X_train[:1000], y_train[:1000]

    return (X_train, y_train), (X_test, y_test), (X_tune, y_tune)

# --- Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.layers(x)

# --- Training & Evaluation ---
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    val_losses = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))
    return val_losses

# --- Optuna Objective ---
def objective(trial, model_template, data, optimizer_name):
    (X_tune, y_tune) = data
    train_loader = TensorDataLoader((X_tune, y_tune), batch_size=32, shuffle=True)

    model = copy.deepcopy(model_template)

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'LRGAdam':
        rank = trial.suggest_int('rank', 2, 20)
        optimizer = LRGAdam(model.parameters(), lr=lr, rank=rank)
    else:
        raise ValueError("Unknown optimizer")

    val_losses = train_and_evaluate(model, optimizer, train_loader, train_loader, epochs=5) # Short run for tuning
    return min(val_losses)

# --- Main Execution ---
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test), (X_tune, y_tune) = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=64)

    model_template = SimpleMLP()

    # Tune Adam
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, model_template, (X_tune, y_tune), 'Adam'), n_trials=15)
    best_lr_adam = study_adam.best_trial.params['lr']
    print(f"Best LR for Adam: {best_lr_adam}")

    # Tune LRGAdam
    study_lrgadam = optuna.create_study(direction='minimize')
    study_lrgadam.optimize(lambda trial: objective(trial, model_template, (X_tune, y_tune), 'LRGAdam'), n_trials=15)
    best_params_lrgadam = study_lrgadam.best_trial.params
    print(f"Best Params for LRG-Adam: {best_params_lrgadam}")

    # Final comparison run
    epochs = 20

    # Adam
    model_adam = copy.deepcopy(model_template)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    adam_losses = train_and_evaluate(model_adam, optimizer_adam, train_loader, test_loader, epochs=epochs)

    # LRGAdam
    model_lrgadam = copy.deepcopy(model_template)
    optimizer_lrgadam = LRGAdam(model_lrgadam.parameters(), lr=best_params_lrgadam['lr'], rank=best_params_lrgadam['rank'])
    lrgadam_losses = train_and_evaluate(model_lrgadam, optimizer_lrgadam, train_loader, test_loader, epochs=epochs)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label='Adam')
    plt.plot(lrgadam_losses, label='LRG-Adam')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Adam vs. LRG-Adam Convergence')
    plt.legend()
    plt.grid(True)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'convergence_comparison.png'))
    print(f"Plot saved to {os.path.join(script_dir, 'convergence_comparison.png')}")
