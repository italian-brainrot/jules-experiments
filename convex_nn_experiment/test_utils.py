import torch
import pytest
from model import ConvexNN
from convexity import hutchinson_hessian_trace
from regularization import sorting_regularization

def test_sorting_regularization():
    """
    Tests the sorting regularization function.
    """
    model = ConvexNN(10, 50, 1)
    # Manually set the weights to be unsorted
    with torch.no_grad():
        model.fc1.weight.copy_(torch.randn_like(model.fc1.weight))
    reg_loss = sorting_regularization(model)
    assert reg_loss > 0

def test_hutchinson_hessian_trace():
    """
    Tests the Hutchinson Hessian trace approximation.
    """
    model = ConvexNN(10, 50, 1)
    data = torch.randn(1, 10)
    target = torch.randn(1, 1)
    criterion = torch.nn.MSELoss()
    trace = hutchinson_hessian_trace(model, data, target, criterion)
    assert isinstance(trace, float)
