import unittest
import torch
import numpy as np
from gaussian_process import GaussianProcess2D, rbf_kernel

class TestGP(unittest.TestCase):

    def setUp(self):
        self.input_file = '../data1.nc'

    def tearDown(self):
        pass

    def test_synthetic_gp(self):

        torch.manual_seed(42)
        n_train = 100
        X_train = torch.rand(n_train, 2) * 10
        y_train = torch.sin(X_train[:, 0]) + torch.cos(X_train[:, 1]) + torch.randn(n_train) * 0.1

        # Create and fit the Gaussian Process model
        gp = GaussianProcess2D(kernel=lambda X1, X2: rbf_kernel(X1, X2, length_scale=2.0, variance=1.0), noise_variance=0.01)
        gp.fit(X_train, y_train)

        # Generate test data
        n_test = 50
        X_test = torch.rand(n_test, 2) * 10

        # Make predictions
        mean, variance = gp.predict(X_test)
        self.assertEqual(X_test.shape, torch.Size([50, 2]))
        self.assertEqual(X_train.shape, torch.Size([100, 2]))
        self.assertEqual(y_train.shape, torch.Size([100]))
        self.assertEqual(mean.shape, torch.Size([50]))
        #print(mean[0], torch.Tensor(-0.2174))
        torch.testing.assert_close(mean[0].item(), 
                                   -0.2174, 
                                   rtol=1.e-4, atol=1.e-4)
        torch.testing.assert_close(variance[0].item(), 
                                   0.0353,
                                   rtol=1.e-4, atol=1.e-4)