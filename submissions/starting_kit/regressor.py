from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adagrad
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):

    def __init__(self):        
        self.reg = Sequential()
        self.reg.add( Dense(50,activation='relu', input_shape= (14,))  ) 
        self.reg.add( Dense(50,activation='relu'))
        self.reg.add( Dense(50,activation='relu'))
        self.reg.add( Dense(50,activation='relu'))
        self.reg.add( Dense(26) )

        self.reg.compile( optimizer=Adagrad() , loss="mse", metrics=["mape"])

        
    def fit(self,X,Y):        
        self.reg.fit( X , Y ,  epochs=100, batch_size=30, verbose=0)

        
    def predict(self,X):
        res = self.reg.predict( X )
        return res
    
