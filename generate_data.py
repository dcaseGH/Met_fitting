''' Make data in particular way for fitting programs'''
import torch
import cf

def generate_data(input_file, n_test=None):
    ''' reads file and returns torch tensors for training, and cfpython data for plotting '''
    # Generalize all of this if it turns out to be useful -  at the moment only works on one variable in one format
    f = cf.read(input_file, select='eastward_wind')[0]
    mydata = f.collapse('mean','longitude') 
    # Sunder training and test data if wanted - Better to apply mask?
    if n_test:

        counter = 0
        x_test, x_train, y_test, y_train = [], [], [], []
        selection_space = int(mydata.size / n_test)
        temp_array = mydata.array[0,:,:,0].transpose().flatten()
        for x in mydata.coord('latitude').array:
            for y in mydata.coord('pressure').array:
                if counter%selection_space == 0:
                    x_test.append([x,y])
                    y_test.append(temp_array[counter])
                else:
                    x_train.append([x,y])
                    y_train.append(temp_array[counter])
                counter +=1
        x_train = torch.tensor(x_train, dtype = torch.float)
        x_test  = torch.tensor(x_test, dtype = torch.float)
        y_train = torch.tensor(y_train, dtype = torch.float).unsqueeze(1)
        y_test  = torch.tensor(y_test, dtype = torch.float).unsqueeze(1)

    else:
        x_train = torch.tensor([[x,y] for x in mydata.coord('latitude').array for y in mydata.coord('pressure').array], dtype = torch.float)
        y_train = torch.tensor(mydata.array[0,:,:,0].transpose().flatten(), dtype = torch.float).unsqueeze(1)
        x_test = None
        y_test = None

    return mydata, x_train, y_train, x_test, y_test