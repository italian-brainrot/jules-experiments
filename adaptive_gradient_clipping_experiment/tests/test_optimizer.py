import torch
import torch.nn as nn
import pytest
from ..optimizer import AdaptiveClippingOptimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def test_optimizer_step():
    model = SimpleModel()
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = AdaptiveClippingOptimizer(base_optimizer)

    # Dummy input and target
    input_tensor = torch.randn(5, 10)
    target_tensor = torch.randint(0, 2, (5,))

    # Forward pass
    output = model(input_tensor)
    loss = nn.CrossEntropyLoss()(output, target_tensor)

    # Backward pass and optimizer step
    loss.backward()
    optimizer.step()

    # Check if gradients are cleared
    optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        assert param.grad is None

if __name__ == "__main__":
    pytest.main()
