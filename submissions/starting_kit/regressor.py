from keras import models
from keras import layers
from keras.optimizers import Adagrad


def Regressor():

    def __init__(self):
        
        self.model = models.Sequential()
        self.model.add( layers.Dense(100,activation='relu', input_shape= (train_data[input_params + ["times"]].shape[1],))  )
        self.model.add( layers.Dense(100,activation='relu'))
        self.model.add( layers.Dense(100,activation='relu'))
        self.model.add( layers.Dense(50,activation='relu'))
        self.model.add( layers.Dense(26) )

        self.model.compile( optimizer=Adagrad() , loss="mse", metrics=["mape"])

    def fit(self):
        
        self.model.fit( train_data[input_params + ["times"]] , train_data[["target_"+i for i in alphabet]] , 
                        epochs=500, batch_size=20, validation_split=0.3,shuffle=True,verbose=0)

    def predict(self):        
        res = self.model.predict( test_data[input_params + ["times"]]  )

        return res
    
