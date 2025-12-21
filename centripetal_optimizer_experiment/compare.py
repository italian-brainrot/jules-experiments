import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import numpy as np
import os
from optimizer import CentripetalOptimizer
import copy

# Define the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Training function
def train(model, data_loader, optimizer, criterion):
    model.train()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Objective function for Optuna
def objective(trial, optimizer_name, train_loader, val_loader, initial_model_state):
    model = MLP(input_size=40, hidden_size=128, output_size=10)
    model.load_state_dict(initial_model_state)
    criterion = nn.CrossEntropyLoss()

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    if optimizer_name == 'Centripetal':
        strength = trial.suggest_float('strength', 1e-5, 1e-1, log=True)
        base_optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = CentripetalOptimizer(base_optimizer, strength=strength)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):  # Short training for hyperparameter tuning
        train(model, train_loader, optimizer, criterion)

    return evaluate(model, val_loader, criterion)

# Main script
if __name__ == "__main__":
    # Load data
    defaults = get_dataset_args()
    defaults.num_samples = 10000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'], dtype=torch.long)
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'], dtype=torch.long)
    train_loader = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
    val_loader = TensorDataLoader((X_test, y_test), batch_size=32)

    # Initial model
    initial_model = MLP(input_size=40, hidden_size=128, output_size=10)
    initial_model_state = copy.deepcopy(initial_model.state_dict())

    # Tune hyperparameters for Adam
    study_adam = optuna.create_study(direction='minimize')
    study_adam.optimize(lambda trial: objective(trial, 'Adam', train_loader, val_loader, initial_model_state), n_trials=20)
    best_lr_adam = study_adam.best_params['lr']

    # Tune hyperparameters for Centripetal Optimizer
    study_centripetal = optuna.create_study(direction='minimize')
    study_centripetal.optimize(lambda trial: objective(trial, 'Centripetal', train_loader, val_loader, initial_model_state), n_trials=20)
    best_lr_centripetal = study_centripetal.best_params['lr']
    best_strength_centripetal = study_centripetal.best_params['strength']

    # Train final models with best hyperparameters
    epochs = 50

    # Adam
    model_adam = MLP(input_size=40, hidden_size=128, output_size=10)
    model_adam.load_state_dict(initial_model_state)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=best_lr_adam)
    criterion = nn.CrossEntropyLoss()
    adam_losses = []
    for epoch in range(epochs):
        train(model_adam, train_loader, optimizer_adam, criterion)
        val_loss = evaluate(model_adam, val_loader, criterion)
        adam_losses.append(val_loss)

    # Centripetal Optimizer
    model_centripetal = MLP(input_size=40, hidden_size=128, output_size=10)
    model_centripetal.load_state_dict(initial_model_state)
    base_optimizer_centripetal = optim.Adam(model_centripetal.parameters(), lr=best_lr_centripetal)
    optimizer_centripetal = CentripetalOptimizer(base_optimizer_centripetal, strength=best_strength_centripetal)
    centripetal_losses = []
    for epoch in range(epochs):
        train(model_centripetal, train_loader, optimizer_centripetal, criterion)
        val_loss = evaluate(model_centripetal, val_loader, criterion)
        centripetal_losses.append(val_loss)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(adam_losses, label=f'Adam (best lr={best_lr_adam:.4f})')
    plt.plot(centripetal_losses, label=f'Centripetal(Adam) (best lr={best_lr_centripetal:.4f}, strength={best_strength_centripetal:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True)

    # Ensure the directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save the plot
    plt.savefig(os.path.join(script_dir, 'comparison.png'))

    print("Comparison complete. Plot saved to comparison.png")
