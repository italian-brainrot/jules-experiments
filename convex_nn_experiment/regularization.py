import torch

def sorting_regularization(model, strength=1.0):
    """
    Computes a regularization term that penalizes unsorted weights.

    Args:
        model (nn.Module): The neural network model.
        strength (float): The strength of the regularization.

    Returns:
        torch.Tensor: The regularization loss.
    """
    reg_loss = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            weights = module.weight
            sorted_weights, _ = torch.sort(weights, dim=1)
            reg_loss += torch.sum((weights - sorted_weights) ** 2)

    return strength * reg_loss
