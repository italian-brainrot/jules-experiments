import torch
import torch.nn as nn
import torch.optim as optim
from light_dataloader import TensorDataLoader
from mnist1d.data import make_dataset, get_dataset_args
import optuna

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def wasserstein_distance_1d(p, q):
    """
    Computes the 1D Wasserstein distance between two probability distributions.
    p and q are tensors of shape (batch_size, num_classes).
    """
    p_cdf = torch.cumsum(p, dim=-1)
    q_cdf = torch.cumsum(q, dim=-1)
    return torch.mean(torch.sum(torch.abs(p_cdf - q_cdf), dim=-1))

def wasserstein_loss(outputs, targets, lambda_reg):
    """
    Computes the composite loss.
    """
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)

    softmax_outputs = torch.softmax(outputs, dim=-1)
    one_hot_targets = nn.functional.one_hot(targets, num_classes=outputs.shape[-1]).float()

    w_dist = wasserstein_distance_1d(softmax_outputs, one_hot_targets)

    return ce_loss + lambda_reg * w_dist

def train(model, data_loader, optimizer, loss_fn, lambda_reg=None):
    model.train()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        if lambda_reg is not None:
            loss = loss_fn(outputs, targets, lambda_reg)
        else:
            loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

# Data loading
defaults = get_dataset_args()
defaults.num_samples = 10000
data = make_dataset(defaults)

X_train, y_train = torch.tensor(data['x'], dtype=torch.float32), torch.tensor(data['y'])
X_test, y_test = torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data['y_test'])

train_loader = TensorDataLoader((X_train, y_train), batch_size=32, shuffle=True)
test_loader = TensorDataLoader((X_test, y_test), batch_size=32, shuffle=False)

input_dim = X_train.shape[1]
output_dim = len(torch.unique(y_train))

def objective_wasserstein(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    lambda_reg = trial.suggest_float("lambda_reg", 1e-5, 1e-1, log=True)

    model = MLP(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        train(model, train_loader, optimizer, wasserstein_loss, lambda_reg)

    return evaluate(model, test_loader)

def objective_baseline(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    model = MLP(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        train(model, train_loader, optimizer, nn.CrossEntropyLoss())

    return evaluate(model, test_loader)

if __name__ == "__main__":
    print("Running Optuna study for Wasserstein regularizer...")
    study_wasserstein = optuna.create_study(direction="maximize")
    study_wasserstein.optimize(objective_wasserstein, n_trials=15)

    print("\nRunning Optuna study for baseline...")
    study_baseline = optuna.create_study(direction="maximize")
    study_baseline.optimize(objective_baseline, n_trials=15)

    print("\n--- Results ---")
    print(f"Best accuracy with Wasserstein regularization: {study_wasserstein.best_value:.4f}")
    print(f"Best params for Wasserstein regularization: {study_wasserstein.best_params}")
    print(f"Best accuracy for baseline: {study_baseline.best_value:.4f}")
    print(f"Best params for baseline: {study_baseline.best_params}")
