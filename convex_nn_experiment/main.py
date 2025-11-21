import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvexNN, BaselineNN
from dataset import get_synthetic_dataset
from convexity import hutchinson_hessian_trace
from regularization import sorting_regularization

def main():
    # Hyperparameters
    input_dim = 10
    hidden_dim = 50
    output_dim = 1
    learning_rate = 0.01
    num_epochs = 100
    reg_strength = 0.01

    # Get the dataset
    X, y = get_synthetic_dataset(n_samples=100, n_features=input_dim)

    # Initialize models
    convex_nn = ConvexNN(input_dim, hidden_dim, output_dim)
    baseline_nn = BaselineNN(input_dim, hidden_dim, output_dim)

    # Loss and optimizers
    criterion = nn.MSELoss()
    convex_optimizer = optim.SGD(convex_nn.parameters(), lr=learning_rate)
    baseline_optimizer = optim.SGD(baseline_nn.parameters(), lr=learning_rate)

    # Training loops
    convex_losses = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = convex_nn(X)
        loss = criterion(outputs, y)
        reg_loss = sorting_regularization(convex_nn, strength=reg_strength)
        total_loss = loss + reg_loss
        convex_losses.append(loss.item())

        # Backward and optimize
        convex_optimizer.zero_grad()
        total_loss.backward()
        convex_optimizer.step()

    baseline_losses = []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = baseline_nn(X)
        loss = criterion(outputs, y)
        baseline_losses.append(loss.item())

        # Backward and optimize
        baseline_optimizer.zero_grad()
        loss.backward()
        baseline_optimizer.step()

    # Calculate convexity measure
    convex_nn_trace = hutchinson_hessian_trace(convex_nn, X, y, criterion)
    baseline_nn_trace = hutchinson_hessian_trace(baseline_nn, X, y, criterion)

    # Print results
    print(f"Convex NN Final Loss: {convex_losses[-1]:.4f}")
    print(f"Baseline NN Final Loss: {baseline_losses[-1]:.4f}")
    print(f"Convex NN Hessian Trace: {convex_nn_trace:.4f}")
    print(f"Baseline NN Hessian Trace: {baseline_nn_trace:.4f}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(convex_losses, label="Convex NN")
    plt.plot(baseline_losses, label="Baseline NN")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.savefig("convex_nn_experiment/loss_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
