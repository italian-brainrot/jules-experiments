import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
from optimizer import GradientAcceleration
import numpy as np
import random

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Load MNIST1D dataset
args = get_dataset_args()
data = get_dataset(args)
X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

# Create DataLoaders
train_loader = TensorDataLoader((X_train, y_train), batch_size=256, shuffle=True)
test_loader = TensorDataLoader((X_test, y_test), batch_size=256, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
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

def train_model(optimizer, train_loader, test_loader, n_epochs=50):
    model = MLP()
    optimizer_instance = optimizer(model.parameters())
    criterion = nn.CrossEntropyLoss()
    val_losses = []

    for epoch in range(n_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer_instance.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer_instance.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                outputs = model(x_val)
                val_loss += criterion(outputs, y_val).item()
        val_losses.append(val_loss / len(test_loader))

    return val_losses

def objective(trial, optimizer_name):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    if optimizer_name == 'Adam':
        optimizer = lambda params: optim.Adam(params, lr=lr)
    else:
        beta = trial.suggest_float('beta', 0.1, 0.99)
        base_optimizer = optim.Adam
        optimizer = lambda params: GradientAcceleration(base_optimizer(params, lr=lr), beta=beta)

    val_losses = train_model(optimizer, train_loader, test_loader, n_epochs=30) # Fewer epochs for tuning
    return min(val_losses)

# --- Optuna Study ---
print("Running Optuna study for Adam...")
study_adam = optuna.create_study(direction='minimize')
study_adam.optimize(lambda trial: objective(trial, 'Adam'), n_trials=20)
best_lr_adam = study_adam.best_params['lr']
print(f"Best LR for Adam: {best_lr_adam}")

print("Running Optuna study for GradientAcceleration(Adam)...")
study_ga = optuna.create_study(direction='minimize')
study_ga.optimize(lambda trial: objective(trial, 'GradientAcceleration'), n_trials=20)
best_lr_ga = study_ga.best_params['lr']
best_beta_ga = study_ga.best_params['beta']
print(f"Best LR for GA(Adam): {best_lr_ga}, Best Beta: {best_beta_ga}")

# --- Final Training ---
print("Training final models...")
adam_optimizer = lambda params: optim.Adam(params, lr=best_lr_adam)
adam_losses = train_model(adam_optimizer, train_loader, test_loader)

ga_optimizer = lambda params: GradientAcceleration(optim.Adam(params, lr=best_lr_ga), beta=best_beta_ga)
ga_losses = train_model(ga_optimizer, train_loader, test_loader)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(adam_losses, label='Adam')
plt.plot(ga_losses, label='Gradient Acceleration (Adam)')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Adam vs. Gradient Acceleration (Adam)')
plt.legend()
plt.grid(True)
plt.savefig('comparison.png')
print("Plot saved to comparison.png")
