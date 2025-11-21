import torch
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def get_synthetic_dataset(n_samples=100, n_features=10, n_informative=5, noise=0.1, random_state=42):
    """
    Generates a synthetic regression dataset.

    Args:
        n_samples (int): The number of samples.
        n_features (int): The number of features.
        n_informative (int): The number of informative features.
        noise (float): The standard deviation of the Gaussian noise applied to the output.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing the input data (X) and target data (y).
    """
    # Generate the dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return X, y
