import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna
import matplotlib.pyplot as plt
import os
import copy

from optimizer import AdaptiveClippingOptimizer

# Define the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load data
def get_data():
    defaults = get_dataset_args()
    defaults.num_samples = 4000
    data = make_dataset(defaults)
    X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])
    train_loader = TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    return train_loader, test_loader, data['x'].shape[1], len(set(data['y']))

# Train and evaluate function
def train_and_evaluate(optimizer, model, train_loader, test_loader, criterion, epochs=10, clip_threshold=None, record_history=False):
    history = []
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if clip_threshold:
                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        if record_history:
            history.append(avg_val_loss)

    return avg_val_loss, history


# Optuna objective function for Adam with fixed clipping
def objective_fixed(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    clip_threshold = trial.suggest_float('clip_threshold', 0.1, 10.0)

    train_loader, test_loader, input_size, num_classes = get_data()
    model = MLP(input_size, 50, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    final_val_loss, _ = train_and_evaluate(optimizer, model, train_loader, test_loader, criterion, clip_threshold=clip_threshold)
    return final_val_loss

# Optuna objective function for Adaptive Clipping Optimizer
def objective_adaptive(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    alpha = trial.suggest_float('alpha', 0.5, 5.0)
    beta = trial.suggest_float('beta', 0.9, 0.999)

    train_loader, test_loader, input_size, num_classes = get_data()
    model = MLP(input_size, 50, num_classes)
    base_optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = AdaptiveClippingOptimizer(base_optimizer, alpha=alpha, beta=beta)
    criterion = nn.CrossEntropyLoss()

    final_val_loss, _ = train_and_evaluate(optimizer, model, train_loader, test_loader, criterion)
    return final_val_loss

if __name__ == '__main__':
    # Tune hyperparameters
    study_fixed = optuna.create_study(direction='minimize')
    study_fixed.optimize(objective_fixed, n_trials=30)

    study_adaptive = optuna.create_study(direction='minimize')
    study_adaptive.optimize(objective_adaptive, n_trials=30)

    best_params_fixed = study_fixed.best_params
    best_params_adaptive = study_adaptive.best_params

    print(f"Best params for fixed clipping: {best_params_fixed}")
    print(f"Best params for adaptive clipping: {best_params_adaptive}")

    # Run final comparison with best hyperparameters
    train_loader, test_loader, input_size, num_classes = get_data()

    # Model for fixed clipping
    model_fixed = MLP(input_size, 50, num_classes)
    optimizer_fixed = optim.Adam(model_fixed.parameters(), lr=best_params_fixed['lr'])
    criterion = nn.CrossEntropyLoss()
    _, history_fixed = train_and_evaluate(optimizer_fixed, model_fixed, train_loader, test_loader, criterion, epochs=20, clip_threshold=best_params_fixed['clip_threshold'], record_history=True)

    # Model for adaptive clipping
    model_adaptive = MLP(input_size, 50, num_classes)
    base_optimizer_adaptive = optim.Adam(model_adaptive.parameters(), lr=best_params_adaptive['lr'])
    optimizer_adaptive = AdaptiveClippingOptimizer(base_optimizer_adaptive, alpha=best_params_adaptive['alpha'], beta=best_params_adaptive['beta'])
    _, history_adaptive = train_and_evaluate(optimizer_adaptive, model_adaptive, train_loader, test_loader, criterion, epochs=20, record_history=True)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(history_fixed, label='Adam with Fixed Clipping')
    plt.plot(history_adaptive, label='Adaptive Clipping Optimizer (Adam)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Comparison of Gradient Clipping Strategies')
    plt.legend()
    plt.grid(True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'comparison.png'))
    print(f"Plot saved to {os.path.join(script_dir, 'comparison.png')}")
