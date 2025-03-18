import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cf
import cfplot as cfp
from generate_data import generate_data

# Control variables
standarize_data = True


class TwoVariableNet(nn.Module):
    def __init__(self):
        super(TwoVariableNet, self).__init__()
        self.hidden1 = nn.Linear(2, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.output(x)
        return x

# Create an instance of the network
net = TwoVariableNet()

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Generate training data
n_test = 20
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



# Training loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4e}')

if n_test:
    # Quick look at test points
    for i in range(n_test):
        print(x_test[i], y_test[i], net(x_test[i]))

# Contour plot of results
pred_y = net(x_test)
fig, ax = plt.subplots()
ax.tricontour(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten(), levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten() , levels=14, cmap="RdBu_r")
fig.colorbar(cntr2, ax=ax)
ax.set_title('Plane')
# Pressure is usually shown with 1000 at the bottom as atmospheric pressure decreases with height
ax.invert_yaxis()
plt.scatter(x_test[:,0], x_test[:,1]) #check appropriate inversion too
#plt.show() 
