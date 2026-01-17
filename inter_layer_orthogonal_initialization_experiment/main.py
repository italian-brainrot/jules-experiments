
import torch
import torch.nn as nn
from mnist1d.data import get_dataset_args, get_dataset
from light_dataloader import TensorDataLoader
import torch.optim as optim
import numpy as np

def inter_layer_orthogonal_initialization(model):
    """
    Applies inter-layer orthogonal initialization to the linear layers of a model.
    This method ensures that the row space of a layer's weight matrix is
    orthogonal to the column space of the preceding layer's weight matrix.
    """
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]

    with torch.no_grad():
        for i in range(len(linear_layers) - 1):
            l1 = linear_layers[i]
            l2 = linear_layers[i+1]

            w1 = l1.weight.data # Shape: (out1, in1)
            w2 = l2.weight.data # Shape: (out2, out1)

            # Get an orthonormal basis for the column space of w1
            q1, _ = torch.linalg.qr(w1)

            # Projector onto the column space of w1
            p1 = q1 @ q1.t()

            # Projector onto the orthogonal complement
            p1_perp = torch.eye(p1.shape[0], device=p1.device) - p1

            # Project the rows of w2 onto the orthogonal complement of Col(w1)
            # w2_new[i, :] = p1_perp @ w2[i, :]
            # This is equivalent to w2_new = w2 @ p1_perp.T = w2 @ p1_perp
            w2_new = w2 @ p1_perp

            # Store original norm
            norm_w2 = torch.linalg.norm(w2)

            # Rescale w2_new to have the same norm as original w2
            norm_w2_new = torch.linalg.norm(w2_new)
            if norm_w2_new > 1e-8: # Avoid division by zero
                w2_new = w2_new * (norm_w2 / norm_w2_new)

            # Assign the new weights back to the second layer
            l2.weight.data.copy_(w2_new)

    return model

# --- Model Definition ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# --- Data Loading ---
def get_dataloaders(batch_size=128):
    args = get_dataset_args()
    data = get_dataset(args, path_to_data='.')

    X_train = torch.from_numpy(data['x']).float()
    y_train = torch.from_numpy(data['y']).long()
    X_test = torch.from_numpy(data['x_test']).float()
    y_test = torch.from_numpy(data['y_test']).long()

    train_loader = TensorDataLoader((X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader((X_test, y_test), batch_size=batch_size)

    return train_loader, test_loader

# --- Training and Evaluation ---
import pandas as pd
import plotly.graph_objects as go

def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, num_epochs=10, return_history=False):
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        history['train_loss'].append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        history['val_loss'].append(epoch_val_loss / len(test_loader))

    final_val_loss = history['val_loss'][-1]

    if return_history:
        return final_val_loss, history
    return final_val_loss

import optuna

def objective(trial, initialization_method):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = SimpleMLP()

    if initialization_method == 'orthogonal':
        model = inter_layer_orthogonal_initialization(model)
    # Kaiming is the default, so no 'else' needed.

    train_loader, test_loader = get_dataloaders()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_loss = train_and_evaluate(model, optimizer, criterion, train_loader, test_loader)

    return val_loss

if __name__ == '__main__':
    # --- Optuna Study for Baseline (Kaiming) ---
    print("Running Optuna study for Kaiming (Baseline)...")
    study_baseline = optuna.create_study(direction='minimize')
    study_baseline.optimize(lambda trial: objective(trial, 'kaiming'), n_trials=20)

    print("\\n--- Optuna Study for Inter-Layer Orthogonal Init ---")
    study_orthogonal = optuna.create_study(direction='minimize')
    study_orthogonal.optimize(lambda trial: objective(trial, 'orthogonal'), n_trials=20)

    # --- Final Training and Visualization ---
    print("\\n--- Retraining best models and generating plot ---")

    # Get best learning rates
    lr_baseline = study_baseline.best_params['lr']
    lr_orthogonal = study_orthogonal.best_params['lr']

    # Initialize models
    torch.manual_seed(42)
    model_baseline = SimpleMLP()
    torch.manual_seed(42)
    model_orthogonal = inter_layer_orthogonal_initialization(SimpleMLP())

    # Dataloaders and criterion
    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    # Optimizers
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=lr_baseline)
    optimizer_orthogonal = optim.Adam(model_orthogonal.parameters(), lr=lr_orthogonal)

    # Train and get history
    _, history_baseline = train_and_evaluate(model_baseline, optimizer_baseline, criterion, train_loader, test_loader, num_epochs=20, return_history=True)
    _, history_orthogonal = train_and_evaluate(model_orthogonal, optimizer_orthogonal, criterion, train_loader, test_loader, num_epochs=20, return_history=True)

    # Create DataFrame for plotting
    df_baseline = pd.DataFrame(history_baseline)
    df_baseline['method'] = 'Kaiming (Baseline)'
    df_orthogonal = pd.DataFrame(history_orthogonal)
    df_orthogonal['method'] = 'Inter-Layer Orthogonal'
    df_results = pd.concat([df_baseline, df_orthogonal]).reset_index(names='epoch')

    # Save results to CSV
    df_results.to_csv('training_history.csv', index=False)

    # Create plot
    fig = go.Figure()
    for method in df_results['method'].unique():
        df_method = df_results[df_results['method'] == method]
        fig.add_trace(go.Scatter(x=df_method['epoch'], y=df_method['val_loss'], mode='lines+markers', name=method))

    fig.update_layout(
        title='Validation Loss Comparison',
        xaxis_title='Epoch',
        yaxis_title='Validation Loss',
        legend_title='Initialization Method'
    )
    fig.write_image("validation_loss_comparison.png")

    # Save trained models
    torch.save(model_baseline.state_dict(), 'model_baseline.pth')
    torch.save(model_orthogonal.state_dict(), 'model_orthogonal.pth')

    print("\\n--- Final Results ---")
    print(f"Best LR for Kaiming: {lr_baseline:.6f} | Final Validation Loss: {df_baseline['val_loss'].iloc[-1]:.4f}")
    print(f"Best LR for Orthogonal: {lr_orthogonal:.6f} | Final Validation Loss: {df_orthogonal['val_loss'].iloc[-1]:.4f}")
    print("\\nPlot saved to validation_loss_comparison.png")
    print("Training history saved to training_history.csv")
    print("Models saved to model_baseline.pth and model_orthogonal.pth")
