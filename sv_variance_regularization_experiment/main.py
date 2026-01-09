
import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
DEVICE = torch.device("cpu")
N_TRIALS = 20
N_EPOCHS = 20
INPUT_DIM = 40
OUTPUT_DIM = 10

# --- Model Definition ---
class MLP(nn.Module):
    def __init__(self, hidden_dims=[128, 128]):
        super(MLP, self).__init__()
        layers = []
        in_dim = INPUT_DIM
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Regularizer ---
def sv_variance_regularizer(model, p=2):
    """
    Computes the variance of the singular values for each weight matrix.
    """
    reg = 0.
    for param in model.parameters():
        if param.dim() > 1:  # Only apply to weight matrices
            s = torch.linalg.svdvals(param)
            reg += torch.var(s)
    return reg

# --- Data Loading ---
def get_data():
    args = get_dataset_args()
    data = get_dataset(args)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

    dl_train = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    dl_test = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return dl_train, dl_test

# --- Training and Evaluation ---
def train_eval(model, dl_train, dl_test, optimizer, criterion, use_regularizer, reg_strength):
    val_losses = []
    for epoch in range(N_EPOCHS):
        model.train()
        for x_batch, y_batch in dl_train:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            if use_regularizer:
                reg_term = sv_variance_regularizer(model)
                loss += reg_strength * reg_term

            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dl_test:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                total_loss += criterion(outputs, y_batch).item()
        val_losses.append(total_loss / len(dl_test))

    return val_losses

# --- Optuna Objective ---
def objective(trial, use_regularizer):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    reg_strength = 0
    if use_regularizer:
        reg_strength = trial.suggest_float("reg_strength", 1e-5, 1e-1, log=True)

    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dl_train, dl_test = get_data()

    val_losses = train_eval(model, dl_train, dl_test, optimizer, criterion, use_regularizer, reg_strength)

    return min(val_losses)

# --- Main Execution ---
if __name__ == "__main__":
    # --- Tune Baseline ---
    study_base = optuna.create_study(direction="minimize")
    study_base.optimize(lambda trial: objective(trial, use_regularizer=False), n_trials=N_TRIALS)
    best_lr_base = study_base.best_params['lr']
    print(f"Best LR for Baseline: {best_lr_base}")

    # --- Tune Regularized Model ---
    study_reg = optuna.create_study(direction="minimize")
    study_reg.optimize(lambda trial: objective(trial, use_regularizer=True), n_trials=N_TRIALS)
    best_lr_reg = study_reg.best_params['lr']
    best_reg_strength = study_reg.best_params['reg_strength']
    print(f"Best LR for Regularized: {best_lr_reg}")
    print(f"Best Reg Strength: {best_reg_strength}")

    # --- Final Comparison ---
    # Baseline
    model_base = MLP().to(DEVICE)
    optimizer_base = torch.optim.Adam(model_base.parameters(), lr=best_lr_base)
    criterion = nn.CrossEntropyLoss()
    dl_train, dl_test = get_data()
    base_losses = train_eval(model_base, dl_train, dl_test, optimizer_base, criterion, False, 0)

    # Regularized
    model_reg = MLP().to(DEVICE)
    # Re-use the same initial weights for a fair comparison
    model_reg.load_state_dict(model_base.state_dict())
    optimizer_reg = torch.optim.Adam(model_reg.parameters(), lr=best_lr_reg)
    reg_losses = train_eval(model_reg, dl_train, dl_test, optimizer_reg, criterion, True, best_reg_strength)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(base_losses, label="Baseline Adam")
    plt.plot(reg_losses, label="Adam with SV Variance Regularization")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Baseline vs. SV Variance Regularization")
    plt.legend()
    plt.grid(True)

    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, "comparison.png"))
    print("Plot saved to comparison.png")

    print(f"Final Baseline Validation Loss: {base_losses[-1]}")
    print(f"Final Regularized Validation Loss: {reg_losses[-1]}")
