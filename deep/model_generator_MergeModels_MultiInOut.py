import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import LeakyReLU
from keras import backend as k
import tensorflow as tf
import keras

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
        keras_backend.set_image_data_format("channels_last")

        rows = data.shape[1]
        columns = data.shape[2]
        samples = len(data)
        channels_cnt = 1

        data = data.reshape(samples, rows, columns, channels_cnt)
        input_shape = (rows, columns, channels_cnt)

        return data, input_shape
    def fit(self, train_xTC, trainxEC, train_y, input_dim, test_x_TC =None, test_x_EC =None, test_y = None): #input_dim example: (600,)
        """ Performs training on train_x instances"""

        train_x_tc, input_shape = self.__reshape_for_cnn(train_xTC)
        test_x_tc , _ = self.__reshape_for_cnn(test_x_TC)

        ''' fucked example '''

        tc_input = Input(shape=input_shape, dtype='float32', name='tc_input')

        tc_model = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu")(tc_input)
        tc_model = MaxPooling2D()(tc_model)
        tc_model = Conv2D(filters=16, kernel_size=(10, 10), strides=(1, 1), padding="same", activation="relu")(tc_model)
        tc_model = MaxPooling2D()(tc_model)
        tc_model = Conv2D(filters=256, kernel_size=(32, 32), strides=(1, 1), padding="same", activation="relu")(tc_model)
        tc_model = MaxPooling2D()(tc_model)


        tc_model = Flatten()(tc_model)
        tc_out = Dense(100, activation = "relu")(tc_model)
        # tc_out = Flatten()(tc_model)

        tc_auxiliary_output = Dense(1, activation='linear', name='tc_auxiliary_output')(tc_out)



        ##Second MOdel
        train_x_ec, input_shape = self.__reshape_for_cnn(trainxEC)
        test_x_ec, _ = self.__reshape_for_cnn(test_x_EC)


        ec_input = Input(shape=input_shape, dtype='float32', name='ec_input')
        ec_model =  Conv2D(filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu")(ec_input)
        ec_model =  MaxPooling2D(pool_size=(1, 5), strides=(1, 5))(ec_model)
        ec_model =  Conv2D(filters=64, kernel_size=(14, 4), strides=1, padding="same", activation="relu")(ec_model)
        ec_model =  MaxPooling2D(pool_size=(1, 4), strides=(1, 4))(ec_model)
        ec_model =  AveragePooling2D(pool_size=(14, 1), strides=(14, 1))(ec_model)
        ec_model =  Conv2D(filters=256, kernel_size=(5, 5), strides=5, padding="same",activation="relu")(ec_model)

        ec_model = Flatten()(ec_model)
        ec_out = Dense(100, activation = "relu")(ec_model)
        # ec_out = Flatten()(ec_model)

        ec_auxiliary_output = Dense(1, activation='linear', name='ec_auxiliary_output')(tc_out)


        tc_model = keras.layers.concatenate([tc_out, ec_out])

        tc_model = Dense(1000, activation='relu')(tc_model)
        tc_model = Dense(500, activation='relu')(tc_model)
        tc_model = Dropout(0.1)(tc_model)
        # tc_model = Dense(100, activation='relu')(tc_model)
        main_output = Dense(1, activation='linear', name='main_output')(tc_model)

        self.__network = Model(inputs=[tc_input, ec_input], outputs=[main_output, tc_auxiliary_output, ec_auxiliary_output])

        adam = optimizers.Adam(lr=self.__learning_rate)
        self.__network.compile(optimizer=adam, loss='mse', loss_weights=[1., 1., 1.], metrics=["accuracy"])

        text_x = [test_x_tc, test_x_ec]

        test_y = [test_y,test_y,test_y]
        self.__history = self.__network.fit({'tc_input': train_x_tc, 'ec_input': train_x_ec},
                  {'main_output': train_y, 'tc_auxiliary_output': train_y, 'ec_auxiliary_output': train_y},
                  epochs=self.__epochs, batch_size=self.__batch_size,
                        verbose=self.__verbose,validation_data=(text_x, test_y))

        ''' fucked example '''


        result = dict()
        result["model"] = self.__network
        result["train_loss_latest"] = 0.0
        result["train_acc_mean"] = 0.0

        result["train_loss"] = 0.0
        print("\nmodel summary:\n--------------")
        print(self.__network.summary())
        return result



    def get_train_loss_mean(self):
        """ get loss and acc during training."""
        if self.__network is not None:
            result = dict()
            result["train_loss_latest"] = 0.0
            result["train_acc_mean"] = 0

            result["train_loss"] = 0.0
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

        # output_activation = self.__activation[len(self.__activation)-1]

        # if  output_activation == "linear":
        predict_values = self.__network.predict(test_x)
        # predict_values = np.array(predict_values[0]).flatten().tolist()
        predict_values = np.array(predict_values[0]).tolist()
        # print(predict_values)
        # print(type(predict_values))

        result["predict"] = predict_values

        # elif output_activation == "softmax":
        #     predict_values = self.__network.predict_classes(test_x)
        #     predict_prob_values = self.__network.predict_proba(test_x)
        #     result["predict"] = predict_values
        #     result["predict_prob"] = predict_prob_values
        #
        # if test_y is not None:
        result["loss_mean"], result["acc_mean"] = (0, 0)

                # self.__network.evaluate({'tc_input': test_x_tc, 'ec_input': test_x_ec},
                #   {'main_output': test_y, 'tc_auxiliary_output': test_y, 'ec_auxiliary_output': test_y}, verbose=0)

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