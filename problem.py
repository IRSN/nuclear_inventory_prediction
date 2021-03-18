import os, pickle, string
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.utils import shuffle
from rampwf.score_types.base import BaseScoreType


problem_title = 'Nuclear inventory of a nuclear reactor core in operation'

_target_names = ["Y_"+ j for j in list(string.ascii_uppercase) ]

Predictions = rw.prediction_types.make_regression( label_names=_target_names )
workflow = rw.workflows.Regressor()


class inventoryError_MSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='this error',precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):        
        mse = ( np.square(y_true - y_pred).sum(axis=1)  ).mean()
        return mse
#----

score_types = [ inventoryError_MSE(name="inventoryError_MSE") ]


def get_train_data(path="."):

    # load pre-prepared dataset aggregating all of the different input data
    # ( for the training dataset, these are composed of 920 different simulation of an operating reactor )
    #train_dataset = pickle.load( open( "./data/train_data_python3.pickle", "rb") )
    train_dataset = pickle.load( open( os.path.join(path,"data","train_data_python3.pickle"), "rb") )

    # Isotopes are named from A to Z 
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0. Those are the input parameters
    # The input parameter space is composed of those initial compositions 
    input_params = alphabet[:8] + ["p%d"%(i) for i in range(1,6) ]

    # We follow the evolution of the composition of the reactor for a total of 81 timesteps 
    timesteps = sorted(list(set(train_dataset["times"])))

    # To use a regression algorithm, we must put the data in the form :
    #
    #   initial composition in A --> H + time  |  output data (compositions from A to Z at T = time)
    #
    # this is done below for the training dataset 
    
    train_data = pd.DataFrame()                   # create a new dataset that will contain the data 
    for simu in range(0,920):                     # loop over each and every simulations (the 920 train data points)
        a = train_dataset.iloc[simu*81:(simu+1)*81]   # slice training dataset (isolate on simulation which have 81 temporal points)  
        b = a[input_params].iloc[0].to_dict()         # get T=0 data (the input data)
        for i in range(1,len(timesteps) ):            # loop over timesteps (from T=20 days to T=1825 days)
            c = a[alphabet + ["times"]]               # get target data ( compos A --> Z at current timestep + time)
            c.columns = ["Y_"+j for j in alphabet] + ["times"]  
            c = c.iloc[i].to_dict()
        
            b.update(c)                               # merge dictionaries having input and output data 
        
            train_data = train_data.append( b , ignore_index=True)  # append current data (input + ouput) into a new dataset
    train_data = shuffle(train_data, random_state=57)  

    return train_data[input_params + ["times"] ].to_numpy() , train_data[ ["Y_"+j for j in alphabet]  ].to_numpy()


def get_test_data(path="."):

    # load pre-prepared dataset aggregating all of the different input data
    # ( for the testing dataset, these are composed of 200 different simulation of an operating reactor )
    test_dataset = pickle.load( open( os.path.join(path,"data","test_data_python3.pickle"), "rb") )

    # Isotopes are named from A to Z 
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0. Those are the input parameters 
    input_params = alphabet[:8]+ ["p%d"%(i) for i in range(1,6) ]

    # We follow the evolution of the composition of the reactor for a total of 81 timesteps 
    timesteps = sorted(list(set(test_dataset["times"])))

    # To use a regression algorithm, we must put the data in the form :
    #
    #   initial composition in A --> H + time  |  output data (compositions from A to Z at T = time)
    #
    # this is done below for the testing dataset 
    
    test_data = pd.DataFrame()                    # create a new dataset that will contain the data 
    for simu in range(0,200):                     # loop over each and every simulations (the 200 test data points)
        a = test_dataset.iloc[simu*81:(simu+1)*81]    # slice testing dataset (isolate on simulation which have 81 temporal points)  
        b = a[input_params].iloc[0].to_dict()         # get T=0 data (the input data)
        for i in range(1,len(timesteps) ):            # loop over timesteps (from T=20 days to T=1825 days)
            c = a[alphabet + ["times"]]               # get target data ( compos A --> Z at current timestep + time)
            c.columns = ["Y_"+j for j in alphabet] + ["times"]  
            c = c.iloc[i].to_dict()
        
            b.update(c)                               # merge dictionaries having input and output data 
        
            test_data = test_data.append( b , ignore_index=True)  # append current data (input + ouput) into a new dataset
    
    test_data = shuffle(test_data, random_state=57)
    
    return  test_data[input_params + ["times"] ].to_numpy() , test_data[ ["Y_"+j for j in alphabet]  ].to_numpy()




def get_cv(X, y):
    return  [(range(0, X.shape[0] -200 ) , range(X.shape[0] -200 , X.shape[0]) ),]
