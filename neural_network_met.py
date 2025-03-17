import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cf
import cfplot as cfp
from generate_data import generate_data

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
mydata, x_train, y_train, x_test, y_test = generate_data('../data1.nc', n_test=20) 
cfp.con(mydata)

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

for i in range(10):
    print(x_train[i], y_train[i], net(x_train[i]))

# Generate testing data
#x_test = torch.linspace(-1, 3, 100).unsqueeze(1)  # Equally spaced points
#y_test = torch.sin(x_test)**2  # f(x) = sin(x)^2
#print(min(x_train[:,0]),max(x_train[:,0]), min(x_train[:,1]),max(x_train[:,1]), 'range points')
xy = np.mgrid[min(x_train[:,0]).numpy():max(x_train[:,0]).numpy():10, min(x_train[:,1]).numpy():max(x_train[:,1]).numpy():50].reshape(2,-1).T
#print(xy.shape, (max(x_train[:,0]).numpy()-min(x_train[:,0]).numpy())/10, 
#                 max(x_train[:,1]).numpy()-min(x_train[:,1]).numpy()/50)
#print(xy[0:10], xy[-1])
z_pred = net(torch.tensor(xy, dtype = torch.float))#.detach().numpy()
#regular grid so can use regular contourplot?
#ax.tricontourf(
X, Y = np.meshgrid(np.arange(min(x_train[:,0]).numpy(), max(x_train[:,0]).numpy(), 10),
                   np.arange(min(x_train[:,1]).numpy(), max(x_train[:,1]).numpy(), 50))
fig, ax = plt.subplots()
c = ax.contourf(X, Y, z_pred.detach().numpy().reshape(X.shape))
fig.colorbar(c, ax=ax)
plt.show()


#matr = np.linspace((min(x_train[:,0]),max(x_train[:,0]), (min(x_train[:,1]),max(x_train[:,1]),10)


#with torch.no_grad():
#    predictions = net(x_test)

# Plot the results
#plt.plot(x_test.numpy(), y_test.numpy(), label='True function', color='r')
#plt.scatter(x_train.numpy(), y_train.numpy(), label='Training data', color='g')
#plt.plot(x_test.numpy(), predictions.numpy(), label='Predictions', linestyle='--')
#plt.legend()
#plt.title('Neural Network Approximation of f(x) = sin(x)^2')
#plt.xlabel('x')
#plt.ylabel('f(x)')
#plt.show()
