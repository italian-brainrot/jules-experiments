import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import os
import copy
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
from layers import KroneckerLinear, LowRankDiagonalLinear, SparseLinear

# Set seed for reproducibility
torch.manual_seed(42)

def get_data():
    args = get_dataset_args()
    args.num_samples = 4000
    # Explicitly set path to avoid conflicts
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist1d_data.pkl')
    # In newer torch, weights_only=True is default. mnist1d uses numpy arrays in pickle.
    # get_dataset uses torch.load internally maybe? No, mnist1d uses pickle.load.
    data = get_dataset(args, path=data_path, download=True)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return (X_train, y_train), (X_test, y_test)

class DenseMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class KroneckerMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, output_dim=10):
        super().__init__()
        # 40 = 5 * 8
        # 100 = 10 * 10
        # 10 = 2 * 5
        self.net = nn.Sequential(
            KroneckerLinear(10, 5, 10, 8), # (10*10) x (5*8) = 100 x 40
            nn.ReLU(),
            KroneckerLinear(2, 10, 5, 10)  # (2*5) x (10*10) = 10 x 100
        )
    def forward(self, x):
        return self.net(x)

class LowRankDiagonalMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, output_dim=10, rank=1):
        super().__init__()
        self.net = nn.Sequential(
            LowRankDiagonalLinear(input_dim, hidden_dim, rank=rank),
            nn.ReLU(),
            LowRankDiagonalLinear(hidden_dim, output_dim, rank=rank)
        )
    def forward(self, x):
        return self.net(x)

class SparseMLP(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=100, output_dim=10, density=0.1):
        super().__init__()
        self.net = nn.Sequential(
            SparseLinear(input_dim, hidden_dim, density=density),
            nn.ReLU(),
            SparseLinear(hidden_dim, output_dim, density=density)
        )
    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, test_loader, lr, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        history['train_loss'].append(total_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        history['test_acc'].append(correct / total)

    return history

def objective(trial, model_class, train_data, input_dim, hidden_dim, output_dim):
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    model = model_class(input_dim, hidden_dim, output_dim)

    X_train, y_train = train_data
    # Use a small subset for faster tuning
    train_loader = TensorDataLoader((X_train[:1000], y_train[:1000]), batch_size=64, shuffle=True)
    val_loader = TensorDataLoader((X_train[1000:1500], y_train[1000:1500]), batch_size=64)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10): # Increased slightly for better signal
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = get_data()
    train_loader = TensorDataLoader((X_train, y_train), batch_size=64, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=64)

    input_dim = 40
    hidden_dim = 100
    output_dim = 10

    models_to_test = {
        'Dense': DenseMLP,
        'Kronecker': KroneckerMLP,
        'LowRankDiag': LowRankDiagonalMLP,
        'Sparse': SparseMLP
    }

    best_lrs = {}
    results = {}
    param_counts = {}

    for name, model_class in models_to_test.items():
        print(f"Tuning {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_class, (X_train, y_train), input_dim, hidden_dim, output_dim), n_trials=10)
        best_lr = study.best_params['lr']
        best_lrs[name] = best_lr
        print(f"Best LR for {name}: {best_lr}")

        print(f"Training {name} with best LR...")
        model = model_class(input_dim, hidden_dim, output_dim)
        param_counts[name] = count_parameters(model)
        history = train_model(model, train_loader, test_loader, best_lr, epochs=30)
        results[name] = history
        print(f"{name} Final Test Acc: {history['test_acc'][-1]:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, history in results.items():
        plt.plot(history['train_loss'], label=f"{name} (params={param_counts[name]})")
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in results.items():
        plt.plot(history['test_acc'], label=f"{name}")
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'comparison_results.png'))
    # plt.show() # Remove show() for headless execution

    # Save summary
    with open(os.path.join(os.path.dirname(__file__), 'summary.txt'), 'w') as f:
        for name in models_to_test:
            f.write(f"{name}: Params={param_counts[name]}, Best LR={best_lrs[name]:.6f}, Final Acc={results[name]['test_acc'][-1]:.4f}\n")
