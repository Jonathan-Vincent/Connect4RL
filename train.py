# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from Connect_4 import Connect_four
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, Add, Input
import json

def load_Q(filename):
    with open(filename, 'r') as f:
        Q = json.load(f)
    return Q

def make_model(inputs, outputs, n_layers=1, n_nodes=64, activation='selu', dropout=0.2):
    input_layer = Input(shape=inputs)
    layer = input_layer
    # make resnet layers
    #for i in range(n_layers):
        #conv_layer = Conv2D(n_nodes, 4, activation=activation, padding='same')(layer)
        #dropout_layer = Dropout(dropout)(conv_layer)
        #adding_layer = Add()([input_layer, dropout_layer])
        #layer = Activation(activation)(conv_layer)
    flatened_layer = Flatten()(layer)
    #policy_dense = Dense(64, activation=activation)(flatened_layer)
    Q_dense = Dense(64, activation=activation)(flatened_layer)
    #output_policy = Dense(outputs[0], activation='softmax')(policy_dense)
    output_Q = Dense(1, activation='tanh')(Q_dense)
    model = keras.Model([input_layer], [output_Q])
    return model

def string_to_state(string):
    return np.array([[1 if x == 'X' else -1 if x == 'O' else 0 for x in row] for row in string.split('\n')]).reshape(6,7)


def pre_process():
    filename = "Q.json"
    Q = load_Q(filename)
    print(len(Q))
    Q = {key: value for key, value in Q.items() if sum(value) > 1}
    print(len(Q))
    Q_list =  list(Q.keys())
    Q_values = list(Q.values())
    Q_arrays = [string_to_state(string) for string in Q_list]
    X = np.array(Q_arrays)
    y = np.array([(Q_values[i][1] - Q_values[i][2])/np.sum(Q_values[i]) for i in range(len(Q_values))])
       
    return X,y


if __name__ == "__main__":
    X,y = pre_process()
    
    model = make_model((6,7,1,), 1)
    model.compile(loss = "mse")
    model.fit(X, y, batch_size = 32, epochs = 20, verbose = 1)    
    
