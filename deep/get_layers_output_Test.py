# this is your initial model
import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import LeakyReLU
from keras import backend as k


import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
import keras.backend as keras_backend


X_Train = np.array([[1, 2, 3], [100, 300, 500]])

Y_Train = np.array([0, 1])  # regression sample


X_Test = np.array([[23, 2321, 245], [10110, 32400, 50130]])


model = Sequential()

model.add(Conv1D(filters = 64, kernel_size=2, padding="same", input_shape=(3, 1)))
model.add(Activation('tanh'))

model.add(Flatten())


model.add(Dense(1))
model.add(Activation('softmax'))

# we train it
model.compile(loss='mse', optimizer='sgd')
X_Train = np.expand_dims(X_Train, axis=2)
print(X_Train.shape)
model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=16)

#
# model = Sequential()
# model.add(Dense(64, input_shape=(3,)))
# model.add(Activation('tanh'))
#
# model.add(Dense(1))
# model.add(Activation('softmax'))
#
# # we train it
# model.compile(loss='mse', optimizer='sgd')
# model.fit(X_Train, Y_Train, nb_epoch=1, batch_size=16)
#
# # we build a new model with the activations of the old model
# # this model is truncated after the first layer
#
# print(np.array(model.layers[0].get_weights()[0]).shape)
# print(np.array(model.layers[0].get_weights()[1]).shape)
# model2 = Sequential()
# model2.add(Dense(64, weights=model.layers[0].get_weights(), input_shape=(3,)))
# model2.add(Activation('tanh'))
#
# activations = model2.predict(X_Test)
# print(activations)