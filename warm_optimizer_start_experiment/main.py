
import torch
import torch.nn as nn
from torch.optim import Adam
from light_dataloader import TensorDataLoader
from mnist1d.data import get_dataset, get_dataset_args

def get_model(input_dim=40, output_dim=10):
    """Creates a simple MLP model."""
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    )

def get_data(num_samples=4000):
    """Fetches and prepares the mnist1d dataset."""
    args = get_dataset_args()
    args.num_samples = num_samples
    data = get_dataset(args, path='./mnist1d_data.pkl', download=False)

    X_train = torch.tensor(data['x'], dtype=torch.float32)
    y_train = torch.tensor(data['y'], dtype=torch.int64)
    X_test = torch.tensor(data['x_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.int64)

    return (
        TensorDataLoader((X_train, y_train), batch_size=128, shuffle=True),
        TensorDataLoader((X_test, y_test), batch_size=128, shuffle=False)
    )

def pretrain_and_get_optimizer_state(lr=1e-3, epochs=10):
    """
    Pre-trains a model to generate an optimizer state dictionary.
    """
    print("Starting pre-training to generate optimizer state...")
    model = get_model()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader, _ = get_data()

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Pre-train Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("Pre-training finished.")
    return optimizer.state_dict()

def train_model(lr, epochs, optimizer_state=None):
    """
    Trains a new, randomly initialized model with either a cold or warm start.
    """
    model = get_model()
    optimizer = Adam(model.parameters(), lr=lr)

    if optimizer_state:
        print("Initializing optimizer with warm start state.")
        # We need to map the state to the new model's parameters.
        # This is a bit tricky because the state is tied to param_groups.
        # A simple load_state_dict won't work if param IDs change.
        # For Adam, we can manually copy 'exp_avg' and 'exp_avg_sq'.

        # Create a dummy optimizer for the new model to get the correct param_ids
        dummy_optimizer = Adam(model.parameters(), lr=lr)
        dummy_optimizer.load_state_dict(optimizer_state)
        optimizer = dummy_optimizer
    else:
        print("Initializing optimizer with cold start.")

    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_data()

    history = []

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        avg_val_loss = total_loss / len(test_loader)
        history.append(avg_val_loss)
        # print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

    return history

if __name__ == '__main__':
    # Example usage:
    print("--- Running Cold Start (Control) ---")
    cold_history = train_model(lr=1e-3, epochs=20, optimizer_state=None)
    print(f"Final validation loss (cold): {cold_history[-1]:.4f}")

    print("\n--- Generating Optimizer State for Warm Start ---")
    warm_state = pretrain_and_get_optimizer_state(lr=1e-3, epochs=10)

    print("\n--- Running Warm Start (Experiment) ---")
    warm_history = train_model(lr=1e-3, epochs=20, optimizer_state=warm_state)
    print(f"Final validation loss (warm): {warm_history[-1]:.4f}")
