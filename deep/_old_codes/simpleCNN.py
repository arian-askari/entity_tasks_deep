import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger
import numpy as np
import keras.backend as keras_backend
X_train = np.array([ [[1,3,4],[1,3,4]],  [[1,3,4],[1,3,4]]  , [[1,3,4],[1,3,4]], [[1,3,4],[1,3,4]] , [[1,3,4],[1,3,4]]  ])
Y_train = np.array([1,0,1,0,1])

keras_backend.set_image_data_format('channels_last')

rows = X_train.shape[1]
columns = X_train.shape[2]
samples = len(X_train)
channels_cnt = 1
X_train = X_train.reshape(samples , rows, columns, channels_cnt)

print(X_train.shape)
print(len(X_train))

print(keras_backend.image_data_format())
#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=(1,1), activation='relu', input_shape=(2,3,1)))
model.add(Conv2D(32, kernel_size=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=3)