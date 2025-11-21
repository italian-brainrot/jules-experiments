import torch
import pytest
from model import ConvexNN, BaselineNN

def test_convex_nn():
    """
    Tests the creation and forward pass of the ConvexNN.
    """
    input_dim = 10
    hidden_dim = 50
    output_dim = 1
    model = ConvexNN(input_dim, hidden_dim, output_dim)
    data = torch.randn(1, input_dim)
    output = model(data)
    assert output.shape == (1, output_dim)

def test_baseline_nn():
    """
    Tests the creation and forward pass of the BaselineNN.
    """
    input_dim = 10
    hidden_dim = 50
    output_dim = 1
    model = BaselineNN(input_dim, hidden_dim, output_dim)
    data = torch.randn(1, input_dim)
    output = model(data)
    assert output.shape == (1, output_dim)
