import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from model import LogisticRegression
from cg_optimizer import ConjugateGradient
from accelerated_cg_optimizer import AcceleratedConjugateGradient

# Load the dataset
data = np.load('cg_curve_prediction/logistic_regression_dataset.npz')
X = data['X']
y = data['y']

# Convert data to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# --- Standard CG ---
model_standard = LogisticRegression(X_tensor.shape[1])
optimizer_standard = ConjugateGradient(model_standard.parameters())
criterion = torch.nn.BCELoss()

def closure_standard():
    optimizer_standard.zero_grad()
    outputs = model_standard(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    return loss

losses_standard = []
start_time_standard = time.time()
for i in range(100):
    loss = optimizer_standard.step(closure_standard)
    losses_standard.append(loss.item())
end_time_standard = time.time()

# --- Accelerated CG ---
model_accelerated = LogisticRegression(X_tensor.shape[1])
optimizer_accelerated = AcceleratedConjugateGradient(model_accelerated.parameters())

def closure_accelerated():
    optimizer_accelerated.zero_grad()
    outputs = model_accelerated(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    return loss

losses_accelerated = []
start_time_accelerated = time.time()
for i in range(100):
    loss = optimizer_accelerated.step(closure_accelerated)
    losses_accelerated.append(loss.item())
end_time_accelerated = time.time()

# --- Comparison ---
plt.figure(figsize=(10, 6))
plt.plot(losses_standard, label='Standard CG')
plt.plot(losses_accelerated, label='Accelerated CG')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Comparison of Standard and Accelerated Conjugate Gradient')
plt.legend()
plt.grid(True)
plt.savefig('cg_curve_prediction/cg_comparison.png')

print(f"Standard CG finished in {end_time_standard - start_time_standard:.4f} seconds.")
print(f"Accelerated CG finished in {end_time_accelerated - start_time_accelerated:.4f} seconds.")
print("Comparison plot saved to 'cg_curve_prediction/cg_comparison.png'")
