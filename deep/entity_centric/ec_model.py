import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import LeakyReLU
import numpy as np
import keras.backend as keras_backend

class Model_Generator():
    def __init__(self, layers, activation=[], optimizer="adam", loss="mse", learning_rate=0.0001,
                 lambd=0, batch_size=100, epochs=1000, dropout=0.0, verbose=2, top_k = 0, category="regression"):
        """
            :param activation : [relu, relu, linear] means, first and second layers have activation relu, and third(output) layer have linear activation function
            :param layers: [1000,1000] means two layers, each layer have 1000 neurons
            :param optimizer: default value, adam
        """

        self.__layers = layers
        self.__activation = activation
        self.__optimizer = optimizer
        self.__loss = loss
        self.__learning_rate = learning_rate
        self.__lambd = lambd  # for L2 regularization
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__dropout = dropout
        self.__verbose = verbose
        self.__category = category
        self.__top_k = top_k
        self.__model_name = self.model_name()
        self.__network = None
        self.__history = None
        self.__csv_log_path = ""
        self.__sess = None
        print("layers =", layers, ", activation =", activation, ", optimizer =", optimizer, ", loss =", loss,
              ", lambda =", lambd, ", learning_rate =", learning_rate, ", batch_size =", batch_size, " ,epochs =", epochs,
              " ,dropout =", dropout, " ,verbose =", verbose, " ,category =", category, " ,top_k=", top_k)

    def set_csv_log_path(self, csv_log_path):
        self.__csv_log_path = csv_log_path

    def  __reshape_for_cnn(self, data, channels_cnt = 1):
        # data = np.rollaxis(data, 1, 4) # roll the axis 1 to position 3,  change channels_first to channels_last ! #https://github.com/keras-team/keras/issues/6598#issuecomment-304615741
        rows = data.shape[1]
        columns = data.shape[2]
        channels_cnt = 1

        samples = len(data)
        data = data.reshape(samples, rows, columns, channels_cnt)
        print("data.shape", data.shape)

        keras_backend.set_image_data_format("channels_last")
        input_shape = (rows, columns, channels_cnt)
        return data, input_shape


    def fit(self, train_x, train_y, input_dim, test_x =None, test_y = None): #input_dim example: (600,)
        """ Performs training on train_x instances"""

        self.__network = Sequential()

        ###################################################################################

        ################################### CNN Part ###################################
        train_x, input_shape = self.__reshape_for_cnn(train_x)
        test_x , _ = self.__reshape_for_cnn(test_x)
        print(input_shape)

        keras_backend.set_image_data_format("channels_last")  # channels_first

        self.__network.add(Conv2D(filters=32, kernel_size= (14,5), strides=1, padding="same", activation="relu", input_shape = input_shape)) #1.2 know terms of entities importancy
        self.__network.add(MaxPooling2D(pool_size=(1,5), strides = (1,5) ))  #2. get max important term five by five

        self.__network.add(Conv2D(filters=64, kernel_size= (14,4), strides=1, padding="same", activation="relu", input_shape = input_shape)) #3 know entities importancy

        self.__network.add(MaxPooling2D(pool_size=(1,4), strides = (1,4) )) #4. get entity iportancy by query phrace
        self.__network.add(AveragePooling2D(pool_size=(14,1), strides = (14,1) )) #5 average of entity iportancy on total query

        self.__network.add(Conv2D(filters=256, kernel_size= (5,5), strides=5, padding="same", activation="relu")) #3 feature reduction
        ################################### CNN Part ###################################

        ################################### Flatt CNN for fed to Dense Layer###################################
        self.__network.add(Flatten())
        ################################### Flatt CNN for fed to Dense Layer###################################


        ################################if full dense, without cnn################################
        # self.__network.add(Dense(self.__layers[0], input_shape=input_dim))
        # self.__network.add(Dropout(self.__dropout))
        # if self.__activation[0] == "LeakyReLU":
        #     self.__network.add(LeakyReLU(alpha=0.2))
        # else:
        #     self.__network.add(Dense(self.__layers[0], input_shape=input_dim))
        # self.__network.add(Dropout(self.__dropout))
        # if self.__activation[0] == "LeakyReLU":
        #     self.__network.add(LeakyReLU(alpha=0.2))
        # else:
        #     self.__network.add(Activation(self.__activation[0]))
        ################################if full dense, without cnn################################

        for i in range(0, len(self.__layers)):
            self.__network.add(Dense(self.__layers[i]))
            if self.__activation[i] == "LeakyReLU":
                self.__network.add(LeakyReLU(alpha=0.2))
            else:
                self.__network.add(Activation(self.__activation[i]))
            self.__network.add(Dropout(self.__dropout))

        if (self.__optimizer == "adam"):
            # adam = optimizers.Adam(lr=self.__learning_rate, decay=0.5)
            adam = optimizers.Adam(lr=self.__learning_rate)
            self.__network.compile(optimizer=adam, loss=self.__loss, metrics=["accuracy"])
        elif(self.__optimizer == "rms"):
            rms_prop = optimizers.RMSprop(lr=self.__learning_rate, rho=0.9, epsilon=None, decay=0.0)
            self.__network.compile(optimizer=rms_prop, loss=self.__loss, metrics=["accuracy"])
        else:
            self.__network.compile(optimizer=self.__optimizer, loss=self.__loss, metrics=["accuracy"])

        if len(self.__csv_log_path) > 0:
            csv_logger = CSVLogger(self.__csv_log_path, append=False, separator=',')

            calbacks = [csv_logger]
            # calbacks = [csv_logger, TerminateOnBaseline(monitor_val='val_loss', monitor_train = 'loss', baseline_min=4.99, baseline_max =5.51)]

            if test_x is not None:
                self.__history = self.__network.fit(train_x, train_y, validation_data=(test_x, test_y),  epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose, callbacks= calbacks)
            else:

                self.__history = self.__network.fit(train_x, train_y, epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose, callbacks=calbacks)
        else:
            if test_x is not None:
                self.__history = self.__network.fit(train_x, train_y, epochs=self.__epochs,
                                                    batch_size=self.__batch_size, verbose=self.__verbose)
            else:
                print(test_y)
                self.__history = self.__network.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose)

        result = dict()
        result["model"] = self.__network
        result["train_loss_latest"] = float(self.__history.history['loss'][-1])
        result["train_acc_mean"] = float(np.mean(self.__history.history['acc']))

        result["train_loss"] = self.__history.history['loss']

        print(self.__network.summary())

        # serialize model to JSON
        model_json = self.__network.to_json()

        with open(self.__csv_log_path + "_model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.__network.save_weights(self.__csv_log_path + "_model.h5")

        print("Saved model to disk")

        return result



    def get_train_loss_mean(self):
        """ get loss and acc during training."""
        if self.__network is not None:
            result = dict()
            result["train_loss_latest"] = float(self.__history.history['loss'][-1])
            result["train_acc_mean"] = float(np.mean(self.__history.history['acc']))

            result["train_loss"] = self.__history.history['loss']
            return self.__network
        else:
            return None

    def get_model_obj(self):
        return self.__network


    def predict(self, test_x, test_y=None):
        """ Performs prediction."""
        test_x , _ = self.__reshape_for_cnn(test_x)
        result = dict()

        output_activation = self.__activation[len(self.__activation)-1]

        if  output_activation == "linear":
            predict_values = self.__network.predict(test_x)
            result["predict"] = predict_values

        elif output_activation == "softmax":
            predict_values = self.__network.predict_classes(test_x)
            predict_prob_values = self.__network.predict_proba(test_x)
            result["predict"] = predict_values
            result["predict_prob"] = predict_prob_values

        if test_y is not None:
            result["loss_mean"], result["acc_mean"] = self.__network.evaluate(test_x, test_y, verbose=0)

        return result

    def model_name(self):
        """
        Generate Name for Model from __init__ parameters
        TODO:: from model summary or from config :)
        :return: model name
        """
        #config = self.__network.get_config()  # TODO:: use this obj for create model name, detailed :)
        model_name = ""
        for i in range(len(self.__layers)):
            model_name += "_L"+ str(i+1) + "(" + str(self.__layers[i]) + ")"

        model_name += "_" + self.__activation[0] +"_drop(" + str(self.__dropout) + ")_opt(" + self.__optimizer + ")"
        model_name += "_epoch(" + str(self.__epochs) + ")" + "_" + "bsize(" + str(self.__batch_size) + ")"
        model_name += "_lr(" + str(self.__learning_rate) + ")"
        model_name += "_lambda(" + str(self.__lambd) + ")"
        model_name += "_loss(" + str(self.__loss) + ")"
        model_name += "_topk(" + str(self.__top_k) + ")"
        model_name += "_" + str(self.__category)
        return model_name

    def get_model_name(self):
        return self.__model_name

    def get_layers(self):
        return self.__layers

    def get_activiation(self):
        return self.__activation

    def get_category(self):
        return self.__category

    def get_dropout(self):
        return self.__dropout

    def get_batch_size(self):
        return self.__batch_size

    def get_epoch_count(self):
        return self.__epochs

    def get_optimizer(self):
        return self.__optimizer

    def get_loss_function(self):
        return self.__loss