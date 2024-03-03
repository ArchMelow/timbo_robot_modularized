'''
code written by Jaejin Lee

defines a simple MLP class
'''
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense
from keras.models import Model
from keras.optimizers import Adam

class MLP(Model):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc0 = Dense(64) # (input_dim, 64)
        self.fc1 = Dense(64) # (64, 64)
        self.fc2 = Dense(output_dim) # (64, output_dim)
        self.act = Activation('relu')
    
    def call(self, x, **kwargs):
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
    
