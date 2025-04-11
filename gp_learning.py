import torch
import numpy as np
import matplotlib.pyplot as plt
import cf
import cfplot as cfp
from generate_data import generate_data
from gaussian_process import GaussianProcess2D, rbf_kernel
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

class RBFKernel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.randn(input_dim))
        self.log_lengthscale = nn.Parameter(torch.randn(1))
        self.log_variance = nn.Parameter(torch.randn(1))

    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / torch.exp(self.log_lengthscale)
        return torch.exp(self.log_variance) * torch.exp(-0.5 * torch.sum(diff ** 2, dim=-1))

class GaussianProcessRegression(nn.Module):
    def __init__(self, X_train, Y_train, kernel):
        super().__init__()
        self.X_train = nn.Parameter(X_train, requires_grad=False)
        self.Y_train = nn.Parameter(Y_train, requires_grad=False)
        self.kernel = kernel
        self.log_noise_variance = nn.Parameter(torch.randn(1))

    def negative_log_likelihood(self):
        K_train = self.kernel(self.X_train) + torch.exp(self.log_noise_variance) * torch.eye(len(self.X_train))
        try:
            L = torch.linalg.cholesky(K_train)
        except torch.linalg.LinAlgError:
            # Add a small diagonal term for numerical stability if Cholesky fails
            K_train = K_train + 1e-6 * torch.eye(len(self.X_train))
            L = torch.linalg.cholesky(K_train)
        alpha = torch.linalg.solve_triangular(L.transpose(-1, -2), torch.linalg.solve_triangular(L, self.Y_train, upper=False), upper=True)
        # note @ symbol is matrix multiplication
        nll = 0.5 * self.Y_train.T @ alpha + torch.sum(torch.log(torch.diag(L))) + 0.5 * len(self.X_train) * torch.log(torch.tensor(2 * np.pi))
        return nll.squeeze()

    def predict(self, X_new):
        K_train = self.kernel(self.X_train) + torch.exp(self.log_noise_variance) * torch.eye(len(self.X_train))
        K_new_train = self.kernel(X_new, self.X_train)
        try:
            L = torch.linalg.cholesky(K_train)
        except torch.linalg.LinAlgError:
            # Add a small diagonal term for numerical stability if Cholesky fails
            K_train = K_train + 1e-6 * torch.eye(len(self.X_train))
            L = torch.linalg.cholesky(K_train)
        alpha = torch.linalg.solve_triangular(L.transpose(-1, -2), torch.linalg.solve_triangular(L, self.Y_train, upper=False), upper=True)
        mean = K_new_train @ alpha

        v = torch.linalg.solve_triangular(L, K_new_train.T, upper=False)
        covariance = self.kernel(X_new) - v.T @ v
        return mean, covariance
# Control variables
standarize_data = True

# Generate training data
n_test = 40
mydata, x_train, y_train, x_test, y_test = generate_data('../data1.nc', n_test=n_test) 
print(mydata)
#cfp.con(mydata)

if standarize_data:# and False:
    # Compute the mean and standard deviation for features/independent variables
    train_mean = x_train.mean(0, keepdim=True)
    train_std_dev = x_train.std(0, unbiased=False, keepdim=True)
    # Standardize the features
    x_train = (x_train - train_mean) / train_std_dev

    if n_test:
        # Same with test data if used
        test_mean = x_test.mean(0, keepdim=True)
        test_std_dev = x_test.std(0, unbiased=False, keepdim=True)
        x_test = (x_test - test_mean) / test_std_dev


# Generate some synthetic 2D data
#np.random.seed(42)
#n_samples = 50
#X_train_np = np.random.rand(n_samples, 2) * 5
#Y_train_np = np.sin(X_train_np[:, 0]) + np.cos(X_train_np[:, 1]) + np.random.randn(n_samples) * 0.2
#X_train = torch.tensor(X_train_np, dtype=torch.float32)
#Y_train = torch.tensor(Y_train_np, dtype=torch.float32).unsqueeze(-1)
#
## Initialize the RBF kernel and the GP model
input_dim = x_train.shape[1]
rbf_kernel = RBFKernel(input_dim)
gp_model = GaussianProcessRegression(x_train, y_train, rbf_kernel)

# Set up the optimizer
optimizer = optim.Adam(gp_model.parameters(), lr=0.1)

# Optimization loop
n_epochs = 50
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = gp_model.negative_log_likelihood()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')


# Contour plot of results
pred_y, var_y = gp_model.predict(x_test)
fig, ax = plt.subplots()
ax.tricontour(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten(), levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten() , levels=14, cmap="RdBu_r")
fig.colorbar(cntr2, ax=ax)
ax.set_title('Plane')
# Pressure is usually shown with 1000 at the bottom as atmospheric pressure decreases with height
ax.invert_yaxis()
plt.scatter(x_test[:,0], x_test[:,1]) #check appropriate inversion too
#plt.savefig('gp_learnt_met.png')

# Make predictions on a grid
#x_min, x_max = X_train_np[:, 0].min() - 1, X_train_np[:, 0].max() + 1
#y_min, y_max = X_train_np[:, 1].min() - 1, X_train_np[:, 1].max() + 1
#xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
#                     np.linspace(y_min, y_max, 50))
#X_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

#with torch.no_grad():
#    mean_pred, covariance_pred = gp_model.predict(X_grid)
#    mean_pred_np = mean_pred.numpy().reshape(xx.shape)
#    std_pred_np = np.sqrt(torch.diag(covariance_pred)).numpy().reshape(xx.shape)

# Plot the results
#plt.figure(figsize=(10, 8))
#plt.scatter(X_train_np[:, 0], X_train_np[:, 1], c=Y_train_np.flatten(), cmap='viridis', label='Training Data')
#contour = plt.contourf(xx, yy, mean_pred_np, cmap='viridis', alpha=0.6)
#plt.colorbar(contour, label='Predicted Mean')
#plt.contour(xx, yy, mean_pred_np, colors='k', linewidths=0.5)
#plt.title('Gaussian Process Regression with Optimized RBF Kernel (2D)')
#plt.xlabel('X1')
#plt.ylabel('X2')
#plt.legend()
#plt.show()

# Print the learned hyperparameters
print("\nLearned Hyperparameters:")
print(f"Lengthscale: {torch.exp(gp_model.kernel.log_lengthscale).detach().numpy()}")
print(f"Variance: {torch.exp(gp_model.kernel.log_variance).detach().numpy()}")
print(f"Noise Variance: {torch.exp(gp_model.log_noise_variance).detach().numpy()}")