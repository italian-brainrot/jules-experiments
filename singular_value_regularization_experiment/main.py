
import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def get_data():
    """Fetches and prepares the mnist1d dataset."""
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.long)
    X_test = torch.tensor(data["x_test"], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)

    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return train_loader, test_loader

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron."""
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

def singular_value_regularization(model, k=1):
    """Computes the sum of the top k singular values for each linear layer's weight."""
    reg = 0.0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Using torch.linalg.svdvals which is more efficient than full SVD
            singular_values = torch.linalg.svdvals(module.weight)
            reg += torch.sum(singular_values[:k])
    return reg

def train_and_evaluate(model, train_loader, test_loader, lr, reg_strength=0.0, epochs=10):
    """Trains and evaluates the model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_losses = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if reg_strength > 0:
                reg_loss = reg_strength * singular_value_regularization(model)
                loss += reg_loss

            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

    return val_losses

def objective(trial, use_regularization):
    """Optuna objective function."""
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    reg_strength = 0.0
    if use_regularization:
        reg_strength = trial.suggest_float("reg_strength", 1e-5, 1e-1, log=True)

    model = MLP()
    train_loader, test_loader = get_data()
    val_losses = train_and_evaluate(model, train_loader, test_loader, lr, reg_strength, epochs=5)
    return min(val_losses)

if __name__ == "__main__":
    # --- Tune Baseline ---
    study_baseline = optuna.create_study(direction="minimize")
    study_baseline.optimize(lambda trial: objective(trial, use_regularization=False), n_trials=10)
    best_lr_baseline = study_baseline.best_params["lr"]

    # --- Tune Regularized Model ---
    study_regularized = optuna.create_study(direction="minimize")
    study_regularized.optimize(lambda trial: objective(trial, use_regularization=True), n_trials=10)
    best_lr_regularized = study_regularized.best_params["lr"]
    best_reg_strength = study_regularized.best_params["reg_strength"]

    print(f"Best LR for Baseline: {best_lr_baseline}")
    print(f"Best LR for Regularized: {best_lr_regularized}")
    print(f"Best Reg Strength for Regularized: {best_reg_strength}")

    # --- Final Training and Comparison ---
    train_loader, test_loader = get_data()

    # Train baseline model
    torch.manual_seed(0) # Reset seed for fair comparison
    np.random.seed(0)
    model_baseline = MLP()
    baseline_losses = train_and_evaluate(model_baseline, train_loader, test_loader, best_lr_baseline, epochs=25)

    # Train regularized model
    torch.manual_seed(0) # Reset seed for fair comparison
    np.random.seed(0)
    model_regularized = MLP()
    regularized_losses = train_and_evaluate(model_regularized, train_loader, test_loader, best_lr_regularized, best_reg_strength, epochs=25)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_losses, label="Baseline MLP")
    plt.plot(regularized_losses, label="MLP with Singular Value Regularization")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Baseline vs. Singular Value Regularization")
    plt.legend()
    plt.grid(True)
    plt.savefig("singular_value_regularization_experiment/comparison.png")
    print("Plot saved to singular_value_regularization_experiment/comparison.png")
