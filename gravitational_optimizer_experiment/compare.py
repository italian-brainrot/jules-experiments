import torch
import torch.nn as nn
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args
from optimizer import GravitationalOptimizer
import optuna
import copy

# --- 1. Dataset Loading ---
def get_data(num_samples=4000, batch_size=128):
    args = get_dataset_args()
    args.num_samples = num_samples
    data = get_dataset(args, force_download=True)
    X_train, y_train = torch.tensor(data['x']).float(), torch.tensor(data['y'])
    X_test, y_test = torch.tensor(data["x_test"]).float(), torch.tensor(data['y_test'])

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# --- 2. Model Definition ---
def get_model(input_dim=40, output_dim=10):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim)
    )

# --- 3. Training and Evaluation Loop ---
def train_eval(model, optimizer, train_loader, test_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples

# --- 4. Optuna Objective Function ---
def objective(trial, model_template, optimizer_name, train_loader, test_loader):
    model = copy.deepcopy(model_template)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "GravitationalOptimizer":
        beta = trial.suggest_float("beta", 0.8, 0.99)
        G = trial.suggest_float("G", 0.1, 10.0, log=True)
        optimizer = GravitationalOptimizer(model.parameters(), lr=lr, beta=beta, G=G)
    else:
        raise ValueError("Unknown optimizer")

    val_loss = train_eval(model, optimizer, train_loader, test_loader)
    return val_loss

# --- 5. Main Execution ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load data and define model template
    train_loader, test_loader = get_data()
    model_template = get_model()

    # --- Tune and Evaluate Adam ---
    print("Tuning Adam...")
    study_adam = optuna.create_study(direction="minimize")
    study_adam.optimize(lambda trial: objective(trial, model_template, "Adam", train_loader, test_loader), n_trials=30)
    best_lr_adam = study_adam.best_params["lr"]

    print(f"Best LR for Adam: {best_lr_adam}")
    final_model_adam = copy.deepcopy(model_template)
    final_optimizer_adam = torch.optim.Adam(final_model_adam.parameters(), lr=best_lr_adam)
    final_loss_adam = train_eval(final_model_adam, final_optimizer_adam, train_loader, test_loader, epochs=20)
    print(f"Final Validation Loss (Adam): {final_loss_adam:.4f}\n")

    # --- Tune and Evaluate GravitationalOptimizer ---
    print("Tuning GravitationalOptimizer...")
    study_gravitational = optuna.create_study(direction="minimize")
    study_gravitational.optimize(lambda trial: objective(trial, model_template, "GravitationalOptimizer", train_loader, test_loader), n_trials=30)
    best_params_gravitational = study_gravitational.best_params

    print(f"Best Params for GravitationalOptimizer: {best_params_gravitational}")
    final_model_gravitational = copy.deepcopy(model_template)
    final_optimizer_gravitational = GravitationalOptimizer(final_model_gravitational.parameters(), **best_params_gravitational)
    final_loss_gravitational = train_eval(final_model_gravitational, final_optimizer_gravitational, train_loader, test_loader, epochs=20)
    print(f"Final Validation Loss (GravitationalOptimizer): {final_loss_gravitational:.4f}\n")

    # --- Final Comparison ---
    print("--- Results ---")
    print(f"Adam Final Validation Loss: {final_loss_adam:.4f}")
    print(f"GravitationalOptimizer Final Validation Loss: {final_loss_gravitational:.4f}")

    if final_loss_gravitational < final_loss_adam:
        print("GravitationalOptimizer performed better.")
    else:
        print("Adam performed better or equal.")
