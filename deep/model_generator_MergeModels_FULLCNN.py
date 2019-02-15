import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import LeakyReLU
from keras import backend as k
import tensorflow as tf


import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
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

    def __reshape_for_cnn(self, data, channels_cnt = 1):
        rows = data.shape[1]
        columns = data.shape[2]
        channels_cnt = 1

        samples = len(data)
        data = data.reshape(samples, rows, columns, channels_cnt)
        print("data.shape", data.shape)

        keras_backend.set_image_data_format("channels_last")
        input_shape = (rows, columns, channels_cnt)
        return data, input_shape

    def fit(self, train_xTC, trainxEC, train_y, input_dim, test_x_TC =None, test_x_EC =None, test_y = None): #input_dim example: (600,)
        """ Performs training on train_x instances"""
        # train_x = np.array(train_x)
        # print(train_x.shape)
        # test_x = np.array(test_x)
        #First model Static
        # print(train_xTC.shape)
        # print(trainxEC.shape)

        train_x_tc, input_shape = self.__reshape_for_cnn(train_xTC)
        test_x_tc , _ = self.__reshape_for_cnn(test_x_TC)
        # train_x_tc, input_shape = self.__reshape_for_cnn(np.array(train_x[0]))
        # test_x_tc , _ = self.__reshape_for_cnn(np.array(test_x[0]))


        model_TC = Sequential()
        model_TC.add(Conv2D(filters=64, kernel_size= (5,5),strides=(1,1), padding="same", activation="relu", input_shape=input_shape, kernel_initializer='uniform'))
        model_TC.add(MaxPooling2D())

        model_TC.add(Conv2D(filters=16, kernel_size= (10,10),strides=(1,1), padding="same", activation="relu"))
        model_TC.add(MaxPooling2D())

        model_TC.add(Conv2D(filters=256, kernel_size= (32,32),strides=(1,1), padding="same", activation="relu"))
        model_TC.add(MaxPooling2D())

        model_TC.add(Flatten())

        model_TC.add(Dense(100))
        model_TC.add(Activation('relu'))
        model_TC.add(Dropout(self.__dropout))

        model_TC.add(Dense(1))
        model_TC.add(Activation('linear'))

        #second try remove it
        # model_TC.add(Dense(100))
        # model_TC.add(Activation('relu'))
        # model_TC.add(Dropout(0.0))



        ##Second MOdel
        train_x_ec, input_shape = self.__reshape_for_cnn(trainxEC)
        test_x_ec, _ = self.__reshape_for_cnn(test_x_EC)

        model_EC = Sequential()

        ##########################################################
        # train_x_ec = np.array(train_x[1])
        # test_x_ec = np.array(test_x[1])
        #
        # model_EC.add(Dense(50, input_shape =(700, )))
        # model_EC.add(Activation('relu'))
        # model_EC.add(Dropout(0))
        ##########################################################


        model_EC.add(Conv2D(filters=32, kernel_size= (5,5), strides=1, padding="same", activation="relu", input_shape = input_shape)) #1.2 know terms of entities importancy
        model_EC.add(MaxPooling2D(pool_size=(1,5), strides = (1,5) ))  #2. get max important term five by five

        model_EC.add(Conv2D(filters=64, kernel_size= (5,4), strides=1, padding="same", activation="relu", input_shape = input_shape)) #3 know entities importancy

        model_EC.add(MaxPooling2D(pool_size=(1,4), strides = (1,4) )) #4. get entity iportancy by query phrace
        model_EC.add(AveragePooling2D(pool_size=(4,1), strides = (4,1) )) #5 average of entity iportancy on total query
        model_EC.add(Flatten())

        model_EC.add(Dense(100))
        model_EC.add(Activation('relu'))


        model_EC.add(Dense(1))
        model_EC.add(Activation('linear'))
        #
        # model_EC.add(Dense(10))
        # model_EC.add(Activation('relu'))
        # model_EC.add(Dropout(self.__dropout))



        #
        # model_EC.add(Dense(10))
        # model_EC.add(Activation('relu'))
        # model_EC.add(Dropout(0))
        #second try remove it
        # model_EC.add(Dense(100))
        # model_EC.add(Activation('relu'))
        # model_EC.add(Dropout(0))

        print(model_TC.summary())
        print(model_EC.summary())


        #Merged Model try to be Dynamic :)
        # model_TC_EC = Add()([model_TC.output, model_EC.output])
        model_TC_EC = Add()([model_TC.layers[8].output, model_EC.layers[7].output])
        # layer[0].get_output()

        ################################NEWWWWWWWWW MErge Stratgey by CNN and conv1D#######################################
        # model_TC_EC = Reshape(200, 1)(model_TC_EC)
        # model_TC_EC = Conv1D(filters=32, kernel_size= 5, strides=1, padding="same", activation="relu", input_shape = input_shape)(model_TC_EC)
        # model_TC_EC = MaxPooling1D(5 , strides=5)(model_TC_EC)
        # model_TC_EC = Conv1D(filters=64, kernel_size= 40, strides=40, padding="same", activation="relu", input_shape = input_shape)(model_TC_EC)
        # model_TC_EC = Flatten()(model_TC_EC)

        ################################NEWWWWWWWWW MErge Stratgey by CNN and conv1D#######################################

        model_TC_EC = Dense(1)(model_TC_EC)
        # model_TC_EC = Activation('relu')(model_TC_EC)
        model_TC_EC = Activation('linear')(model_TC_EC)

        merged_TC_EC_new_model = Sequential()
        merged_TC_EC_new_model = Model([model_TC.input, model_EC.input], model_TC_EC)

        self.__network = merged_TC_EC_new_model
        #Merged Model try to be Dynamic :)
        # print(self.__network.summary())

        train_x  = [train_x_tc, train_x_ec]
        test_x = [test_x_tc, test_x_ec]

        ##########layers added static !
        # for i in range(0, len(self.__layers)):
        #     self.__network.add(Dense(self.__layers[i]))
        #     if self.__activation[i] == "LeakyReLU":
        #         self.__network.add(LeakyReLU(alpha=0.2))
        #     else:
        #         self.__network.add(Activation(self.__activation[i]))
        #     self.__network.add(Dropout(self.__dropout))
        ##########layers added static !

        # arrays = 2
        # rows = train_x.shape[1]
        # columns = train_x.shape[2]
        # channels_cnt = 1
        # input_shape = (2, rows, columns,1)


        if (self.__optimizer == "adam"):
            adam = optimizers.Adam(lr=self.__learning_rate)
            self.__network.compile(optimizer=adam, loss=self.__loss, metrics=["accuracy"])
        elif(self.__optimizer == "rms"):
            rms_prop = optimizers.RMSprop(lr=self.__learning_rate, rho=0.9, epsilon=None, decay=0.0)
            self.__network.compile(optimizer=rms_prop, loss=self.__loss, metrics=["accuracy"])
        else:
            self.__network.compile(optimizer=self.__optimizer, loss=self.__loss, metrics=["accuracy"])

        if len(self.__csv_log_path) > 0:
            csv_logger = CSVLogger(self.__csv_log_path, append=False, separator=',')

            if test_x is not None:
                self.__history = self.__network.fit(train_x, train_y, validation_data=(test_x, test_y),  epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose, callbacks=[csv_logger])
            else:

                self.__history = self.__network.fit(train_x, train_y, epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose, callbacks=[csv_logger])
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
        # print("\nmodel summary:\n--------------")
        print(self.__network.summary())


        # outputs = [layer.output for layer in self.__network.layers]  # all layer outputs
        # for i in range(len(outputs)):
        #     print(i)
        #     # tf.Session.run(outputs[i].output)
        #
        #     print(outputs[i].get_output())
        #
        # kkkkk =0
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


    def predict(self, test_x_TC, test_x_EC, test_y=None):
        """ Performs prediction."""

        test_x_tc , _ = self.__reshape_for_cnn(test_x_TC)
        test_x_ec, _ = self.__reshape_for_cnn(test_x_EC)

        ###########
        # test_x_ec = np.array(test_x[1])
        ###########


        test_x = [test_x_tc, test_x_ec]
        # test_x , _ = self.__reshape_for_cnn(test_x)
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




# NDCG Scorer function


def example():
    # layers = [100, 100, 1]  # regression sample
    layers = [100, 100, 2]  # classification sample

    # activation =  ["relu", "relu", "linear"]
    activation =  ["relu", "relu", "softmax"]

    X_Train = np.array([[1, 2, 3], [100, 300, 500]])

    # Y_Train = np.array([0, 1])  # regression sample
    Y_Train = np.array([[1, 0], [0, 1]])  # classification sample

    X_Test = np.array([ [2, 4, 6], [200, 600, 1000]])

    # Y_Test = np.array([0, 1])  # regression sample
    Y_Test = np.array([[1, 0], [0, 1]])  # classification sample

    input_dim = (3,)

    # model = Model_Generator(layers=layers, activation=activation, epochs=100, dropout=0.4, category="classification")  # classification sample
    model = Model_Generator(layers=layers, activation=activation, epochs=100, dropout=0.4, category="regression")  # regression sample

    result = model.fit(X_Train, Y_Train, input_dim)
    predict_result = model.predict(X_Test, Y_Test)

    print(model.get_model_name())
    print("\n\n--------------------------\n\n\n")

    print(result)
    print("\n\n--------------------------\n\n\n")
    print(predict_result)

    model.get_model_name()

# example()  #uncomment it for see how to work with this helper :)