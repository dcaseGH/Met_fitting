'''
Get some meteorological data and fit a plane through it
Plot the data with cf contour plot and also the best fit plane
'''

seed_for_reproducibility = 3333
import numpy as np
np.random.seed(seed_for_reproducibility)
import random
random.seed(seed_for_reproducibility)
from matplotlib import pyplot as plt
import cf
import cfplot as cfp

import torch
torch.manual_seed(seed_for_reproducibility)

# data from eg online
f = cf.read('../data1.nc', select='eastward_wind')[0]

mydata = f.collapse('mean','longitude') 

#original data
print(mydata)
cfp.con(mydata)

# Reshape 
x0_list = mydata.coord('latitude').array
x1_list = mydata.coord('pressure').array
y_list = mydata.array[0,:,:,0].transpose().flatten()

x0_min, x1_min, y_min, x0_max, x1_max, y_max = min(x0_list) , min(x1_list) , min(y_list) , max(x0_list) , max(x1_list) , max(y_list)
print('\nFollowing are the extreme ends of the synthetic data points...')
print('x0_min, x0_max: {}, {}'.format(x0_min, x0_max))
print('x1_min, x1_max: {}, {}'.format(x1_min, x1_max))
print('y_min, y_max: {}, {}'.format(y_min, y_max))
# Defining the model architecture.
class LinearRegressionModel(torch.nn.Module): 
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(2, 1)  # this layer of the model has a single neuron, that takes in vector 2 in and gives out one scalar output. 
  
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred 

# Creating the model
model = LinearRegressionModel()

print(model)
for name, parameter in model.named_parameters():
    print('name           : {}'.format(name))
    print('parameter      : {}'.format(parameter))#.item()))
#    print('learnable      : {}'.format(parameter.requires_grad))
#    print('parameter.shape: {}'.format(parameter.shape))
    print('---------------------------------')
    
# Defining the Loss Function
# Mean Squared Error is the most common choice of Loss Function for Linear Regression models.
criterion = torch.nn.MSELoss()

# Defining the Optimizer, which would update all the trainable parameters of the model, making the model learn the data distribution better and hence fit the distribution better.
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005) 

# We also need to convert all the data into tensors before we could use them for training our model.
zipped_grid = np.array([[x,y]   for x in mydata.coord('latitude').array for y in mydata.coord('pressure').array])
data_x = torch.tensor([x for x in zipped_grid], dtype = torch.float)
data_y = torch.tensor([[y] for y in y_list], dtype = torch.float)

print('dhc ', data_x.shape, data_y.shape, data_x[0], data_x[2])

losses = []         # to keep track of the epoch lossese 
slope_list = []     # to keep track of the slope learnt by the model
intercept_list = [] # to keep track of the intercept learnt by the model

EPOCHS = 2500
print('\nTRAINING...')
for epoch in range(EPOCHS):
    # We need to clear the gradients of the optimizer before running the back-propagation in PyTorch
    optimizer.zero_grad() 
    
    # Feeding the input data in the model and getting out the predictions
    pred_y = model(data_x)

    # Calculating the loss using the model's predictions and the real y values
    loss = criterion(pred_y, data_y) 

    # Back-Propagation
    loss.backward() 
    
    # Updating all the trainable parameters
    optimizer.step()
    
    # Appending the loss.item() (a scalar value)
    losses.append(loss.item())
    
    # Appending the learnt slope and intercept   
#    slope_list.append(model.linear.weight.item())
#    intercept_list.append(model.linear.bias.item())
    
    # We print out the losses after every 2000 epochs
    if (epoch)%100 == 0:
        print('loss: ', loss.item())

# Contour plot of the fit
fig, ax = plt.subplots()
ax.tricontour(zipped_grid[:,0], zipped_grid[:,1], pred_y.detach().numpy().flatten(), levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(zipped_grid[:,0], zipped_grid[:,1], pred_y.detach().numpy().flatten() , levels=14, cmap="RdBu_r")
fig.colorbar(cntr2, ax=ax)
ax.set_title('Plane')

plt.show()
