import torch
import numpy as np
import matplotlib.pyplot as plt
import cf
import cfplot as cfp
from generate_data import generate_data
from gaussian_process import GaussianProcess2D, rbf_kernel
# Control variables
standarize_data = True

# Generate training data
n_test = 40
mydata, x_train, y_train, x_test, y_test = generate_data('../data1.nc', n_test=n_test) 
print(mydata)
cfp.con(mydata)

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


# The generate data function returns the extra dim for NNs
y_test = y_test.squeeze()
y_train = y_train.squeeze()


# Create an instance of the network
gp = GaussianProcess2D(kernel=lambda X1, X2: rbf_kernel(X1, X2, 
                                                        length_scale=2.0, 
                                                        variance=1.0), 
                                                        noise_variance=0.01)
gp.fit(x_train, y_train)

pred_y, var_y = gp.predict(x_test)

if n_test:
    # Quick look at test points
    for i in range(n_test):
        print(x_test[i], y_test[i], pred_y[i])

# Contour plot of results
fig, ax = plt.subplots()
ax.tricontour(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten(), levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten() , levels=14, cmap="RdBu_r")
fig.colorbar(cntr2, ax=ax)
ax.set_title('Plane')
# Pressure is usually shown with 1000 at the bottom as atmospheric pressure decreases with height
ax.invert_yaxis()
plt.scatter(x_test[:,0], x_test[:,1]) #check appropriate inversion too
#plt.show() 
