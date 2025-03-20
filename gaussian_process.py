import torch
import numpy as np

class GaussianProcess2D(torch.nn.Module):
    """
    A Gaussian Process model for 2D data using PyTorch.
    """

    def __init__(self, kernel, noise_variance=1e-6):
        """
        Initializes the Gaussian Process model.

        Args:
            kernel: A callable that computes the covariance between two points.
            noise_variance: The variance of the observation noise.
        """
        super(GaussianProcess2D, self).__init__()
        self.kernel = kernel
        self.noise_variance = torch.tensor(noise_variance)
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky decomposition of K + noise
        self.device = None

 #   def to_device(self, device=None):
 #       if device:
 #           self.X_train = self.X_train.to(device)
 #           self.y_train = self.y_train.to(device)
 #           self.noise_variance = self.noise_variance.to(device)

    def fit(self, X_train, y_train, device = None):
        """
        Fits the Gaussian Process model to the training data.

        Args:
            X_train: A torch tensor of shape (n_samples, 2) representing the training input points.
            y_train: A torch tensor of shape (n_samples,) representing the training target values.
        """
        self.X_train = X_train
        self.y_train = y_train
        local_unit = torch.eye(len(X_train))
        if device:
            self.device = device
            local_unit = local_unit.to(device)
            self.X_train = self.X_train.to(device)
            self.y_train = self.y_train.to(device)
            self.noise_variance = self.noise_variance.to(device)

        K_train = self.kernel(self.X_train, self.X_train) + self.noise_variance * local_unit
        
        if device:
            self.L = torch.linalg.cholesky(K_train).to(device)
        else:
            self.L = torch.linalg.cholesky(K_train)

    def predict(self, X_test, device = None):
        """
        Predicts the target values for the test input points.

        Args:
            X_test: A torch tensor of shape (n_test_samples, 2) representing the test input points.

        Returns:
            A tuple containing the predicted mean and variance.
        """
        if self.X_train is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Ensure on the same device if not already copied
        if self.device:
            X_test = X_test.to(self.device)
        K_test_train = self.kernel(X_test, self.X_train)
        K_test_test = self.kernel(X_test, X_test)
        
        alpha = torch.linalg.solve(self.L.T, torch.linalg.solve(self.L, self.y_train.unsqueeze(-1))).squeeze(-1) #Added unsqueeze and squeeze to handle vector shapes.
        mean = torch.matmul(K_test_train, alpha)

        v = torch.linalg.solve(self.L, K_test_train.T)
        covariance = K_test_test - torch.matmul(v.T, v)
        variance = torch.diag(covariance)

        return mean, variance

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    """
    Radial Basis Function (RBF) kernel.

    Args:
        X1: A torch tensor of shape (n_samples1, 2).
        X2: A torch tensor of shape (n_samples2, 2).
        length_scale: The length scale parameter.
        variance: The variance parameter.

    Returns:
        A torch tensor of shape (n_samples1, n_samples2) representing the covariance matrix.
    """
    dist = torch.cdist(X1, X2, p=2)**2 #changed to torch.cdist and squared.
    return variance * torch.exp(-dist / (2 * length_scale**2))