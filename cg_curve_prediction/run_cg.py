import numpy as np
import torch
import matplotlib.pyplot as plt
from model import LogisticRegression
from cg_optimizer import ConjugateGradient

# Load the dataset
data = np.load('cg_curve_prediction/logistic_regression_dataset.npz')
X = data['X']
y = data['y']

# Convert data to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Initialize the model
model = LogisticRegression(X_tensor.shape[1])
optimizer = ConjugateGradient(model.parameters())
criterion = torch.nn.BCELoss()

def closure():
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    return loss

trajectories = []
print("Starting Conjugate Gradient optimization...")
for i in range(100):
    loss = optimizer.step(closure)
    print(f"Iteration {i+1}, Loss: {loss.item():.4f}")

    # Store the parameters
    flat_params = optimizer._gather_flat_params().clone().numpy()
    trajectories.append(flat_params)

print("Conjugate Gradient optimization finished.")

# Save the trajectories
trajectories = np.array(trajectories)
np.save('cg_curve_prediction/cg_trajectories.npy', trajectories)

# Plot the trajectories
plt.figure(figsize=(12, 8))
for i in range(trajectories.shape[1]):
    plt.plot(trajectories[:, i], label=f'Param {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameter Trajectories during Conjugate Gradient Optimization')
plt.legend()
plt.grid(True)
plt.savefig('cg_curve_prediction/cg_trajectories.png')
print("Parameter trajectories saved to 'cg_curve_prediction/cg_trajectories.png' and 'cg_curve_prediction/cg_trajectories.npy'")
