
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mnist1d.data import make_dataset, get_dataset_args
from light_dataloader import TensorDataLoader
import optuna
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_data(num_samples=10000, **kwargs):
    args = get_dataset_args()
    args.num_samples = num_samples
    for k, v in kwargs.items():
        setattr(args, k, v)
    data = make_dataset(args)
    X = torch.tensor(data['x']).float()
    y = torch.tensor(data['y']).long()
    X_test = torch.tensor(data['x_test']).float()
    y_test = torch.tensor(data['y_test']).long()
    return X, y, X_test, y_test

def align_covariance(X_source, X_target):
    """
    Aligns the covariance and mean of X_source to X_target using CORAL.
    """
    mu_s = X_source.mean(dim=0)
    X_s_centered = X_source - mu_s
    cov_s = (X_s_centered.T @ X_s_centered) / (X_source.shape[0] - 1)

    mu_t = X_target.mean(dim=0)
    X_t_centered = X_target - mu_t
    cov_t = (X_t_centered.T @ X_t_centered) / (X_target.shape[0] - 1)

    eps = 1e-5
    cov_s += torch.eye(X_source.shape[1]) * eps
    cov_t += torch.eye(X_target.shape[1]) * eps

    L_s, V_s = torch.linalg.eigh(cov_s)
    L_s = torch.clamp(L_s, min=eps)
    cov_s_inv_sqrt = V_s @ torch.diag(1.0 / torch.sqrt(L_s)) @ V_s.T

    L_t, V_t = torch.linalg.eigh(cov_t)
    L_t = torch.clamp(L_t, min=eps)
    cov_t_sqrt = V_t @ torch.diag(torch.sqrt(L_t)) @ V_t.T

    X_aligned = (X_s_centered @ cov_s_inv_sqrt @ cov_t_sqrt) + mu_t
    return X_aligned

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, lr, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dl = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dl:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            if torch.isnan(loss): return
            loss.backward()
            optimizer.step()

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def run_experiment():
    set_seed(42)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    print("Loading ID data...")
    X_train_id, y_train_id, X_test_id, y_test_id = get_data()

    def objective(trial):
        set_seed(42)
        idx = torch.randperm(len(X_train_id))
        train_idx = idx[:4000]
        val_idx = idx[4000:5000]
        X_t, y_t = X_train_id[train_idx], y_train_id[train_idx]
        X_v, y_v = X_train_id[val_idx], y_train_id[val_idx]

        hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)

        model = MLP(40, hidden_dim, 10, n_layers)
        train_model(model, X_t, y_t, lr, epochs=20)
        return evaluate(model, X_v, y_v)

    print("Tuning hyperparameters...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"Best params: {best_params}")

    print("Training best model...")
    model = MLP(40, best_params['hidden_dim'], 10, best_params['n_layers'])
    train_model(model, X_train_id, y_train_id, best_params['lr'], epochs=50)
    acc_id = evaluate(model, X_test_id, y_test_id)
    print(f"ID Accuracy: {acc_id:.4f}")

    scenarios = {
        "High Noise": {"corr_noise_scale": 0.5},
        "High Shear": {"shear_scale": 1.5},
        "Low Scale": {"scale_coeff": 0.2},
        "Translation": {"max_translation": 60},
        "Combined": {"corr_noise_scale": 0.4, "shear_scale": 1.2, "scale_coeff": 0.3}
    }

    exp_results = []

    for name, kwargs in scenarios.items():
        print(f"Running scenario: {name}")
        _, _, X_test_ood, y_test_ood = get_data(**kwargs)

        acc_ood = evaluate(model, X_test_ood, y_test_ood)
        X_test_ood_aligned = align_covariance(X_test_ood, X_train_id)
        acc_ood_aligned = evaluate(model, X_test_ood_aligned, y_test_ood)

        exp_results.append((name, acc_ood, acc_ood_aligned))

        if name == "Combined":
            print("Generating PCA plot...")
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_id[:500].numpy())
            X_test_ood_pca = pca.transform(X_test_ood[:500].numpy())
            X_test_aligned_pca = pca.transform(X_test_ood_aligned[:500].numpy())

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5)
            plt.title("Train (ID)")
            plt.subplot(1, 3, 2)
            plt.scatter(X_test_ood_pca[:, 0], X_test_ood_pca[:, 1], alpha=0.5, color='orange')
            plt.title("Test (OOD)")
            plt.subplot(1, 3, 3)
            plt.scatter(X_test_aligned_pca[:, 0], X_test_aligned_pca[:, 1], alpha=0.5, color='green')
            plt.title("Test (Aligned)")
            plt.tight_layout()
            plt.savefig(os.path.join(dir_path, "pca_comparison.png"))

    print("\nSummary of Results:")
    print(f"ID Accuracy: {acc_id:.4f}")
    print("| Scenario | OOD Accuracy | Aligned OOD Accuracy | Improvement |")
    print("| --- | --- | --- | --- |")
    for name, ood, aligned in exp_results:
        print(f"| {name} | {ood:.4f} | {aligned:.4f} | {aligned-ood:+.4f} |")

if __name__ == "__main__":
    run_experiment()
