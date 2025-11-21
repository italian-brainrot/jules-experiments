import torch
import torch.nn as nn

class ConvexNN(nn.Module):
    """
    A neural network with sorted weights, designed to have a more convex loss landscape.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvexNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class BaselineNN(nn.Module):
    """
    A standard neural network for comparison.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaselineNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
