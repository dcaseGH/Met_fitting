import torch
import numpy as np
import matplotlib.pyplot as plt
import cf
import cfplot as cfp
from generate_data import generate_data
from gaussian_process import GaussianProcess2D, rbf_kernel
from torch.profiler import profile, record_function, ProfilerActivity

# Turn off all plotting and profile main ML code

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


# The generate data function returns the extra dim for NNs
y_test = y_test.squeeze()
y_train = y_train.squeeze()

#prof.export_chrome_trace("trace.json")


if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    print('Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices')
    import sys
    sys.exit(0)

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
sort_by_keyword = device + "_time_total"
print(sort_by_keyword)
gp = GaussianProcess2D(kernel=lambda X1, X2: rbf_kernel(X1, X2, 
                                                        length_scale=2.0, 
                                                        variance=1.0), 
                                                        noise_variance=0.01).to(device)

# Ensure all objects explicitly transferred to device - do this within model object

with profile(activities=activities) as prof:
    gp.fit(x_train, y_train, device=device)
    pred_y, var_y = gp.predict(x_test)

prof.export_chrome_trace("trace.json")
#print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=100))

'''
# See where memory allocated/released - get footprint of model
with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:

    gp = GaussianProcess2D(kernel=lambda X1, X2: rbf_kernel(X1, X2, 
                                                            length_scale=2.0, 
                                                            variance=1.0), 
                                                            noise_variance=0.01)
    gp.fit(x_train, y_train)

    pred_y, var_y = gp.predict(x_test)


print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
'''
'''
# Create an instance of the network and profile the fitting
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("profile_of_gp"):

        gp = GaussianProcess2D(kernel=lambda X1, X2: rbf_kernel(X1, X2, 
                                                                length_scale=2.0, 
                                                                variance=1.0), 
                                                                noise_variance=0.01)
        gp.fit(x_train, y_train)

        pred_y, var_y = gp.predict(x_test)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
'''


if n_test:
    # Quick look at test points
    for i in range(n_test):
        print(x_test[i], y_test[i], pred_y[i])

# Contour plot of results
#fig, ax = plt.subplots()
#ax.tricontour(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten(), levels=14, linewidths=0.5, colors='k')
#cntr2 = ax.tricontourf(x_test[:,0], x_test[:,1], pred_y.detach().numpy().flatten() , levels=14, cmap="RdBu_r")
#fig.colorbar(cntr2, ax=ax)
#ax.set_title('Plane')
# Pressure is usually shown with 1000 at the bottom as atmospheric pressure decreases with height
#ax.invert_yaxis()
#plt.scatter(x_test[:,0], x_test[:,1]) #check appropriate inversion too
#plt.show() 
