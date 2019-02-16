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
        self.__network_TC = None
        self.__network_EC = None
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
        # train_x = np.array(train_x)
        # print(train_x.shape)
        # test_x = np.array(test_x)
        #First model Static
        # print(train_xTC.shape)
        # print(trainxEC.shape)

        two_inner_models_epochs = 100

        train_x_tc, input_shape = self.__reshape_for_cnn(train_xTC)
        test_x_tc , _ = self.__reshape_for_cnn(test_x_TC)
        # train_x_tc, input_shape = self.__reshape_for_cnn(np.array(train_x[0]))
        # test_x_tc , _ = self.__reshape_for_cnn(np.array(test_x[0]))

        model_TC = Sequential()
        model_TC.add(Conv2D(filters=64, kernel_size= (5,5),strides=(1,1), padding="same", activation="relu", input_shape=input_shape))
        model_TC.add(MaxPooling2D())

        model_TC.add(Conv2D(filters=16, kernel_size= (10,10),strides=(1,1), padding="same", activation="relu"))
        model_TC.add(MaxPooling2D())

        model_TC.add(Conv2D(filters=256, kernel_size= (32,32),strides=(1,1), padding="same", activation="relu"))
        model_TC.add(MaxPooling2D())

        model_TC.add(Flatten())

        model_TC.add(Dense(100))
        model_TC.add(Activation('relu'))

        model_TC.add(Dense(1))
        model_TC.add(Activation('linear'))

        tc_learning_rate = 0.0001
        adam = optimizers.Adam(lr=tc_learning_rate)
        model_TC.compile(optimizer=adam, loss="mse", metrics=["accuracy"])

        print("\n\nModel Type Centric Fitting")

        model_TC.fit(train_x_tc, train_y, validation_data=(test_x_tc, test_y), epochs=two_inner_models_epochs,
                           batch_size=128, verbose=2)

        # print(model_TC.summary())
        print("\n\nModel Type Centric Fitted")

        # print(np.array(model_TC.layers[8].get_weights()[0]).shape)
        # print(np.array(model_TC.layers[8].get_weights()[1]).shape)

        self.__network_TC = Sequential()
        self.__network_TC.add(Conv2D(filters=64, weights=model_TC.layers[0].get_weights(), kernel_size= (5,5),strides=(1,1), padding="same", activation="relu", input_shape=input_shape))
        self.__network_TC.add(MaxPooling2D(weights=model_TC.layers[1].get_weights()))
        self.__network_TC.add(Conv2D(weights= model_TC.layers[2].get_weights(),filters=16, kernel_size= (10,10),strides=(1,1), padding="same", activation="relu"))
        self.__network_TC.add(MaxPooling2D(weights=model_TC.layers[3].get_weights()))
        self.__network_TC.add(Conv2D(weights=model_TC.layers[4].get_weights(),filters=256, kernel_size= (32,32),strides=(1,1), padding="same", activation="relu"))
        self.__network_TC.add(MaxPooling2D(weights=model_TC.layers[5].get_weights()))
        self.__network_TC.add(Flatten(weights=model_TC.layers[6].get_weights()))
        # self.__network_TC.add(Dense(100, weights=model_TC.layers[7].get_weights()))
        # self.__network_TC.add(Activation('relu', weights=model_TC.layers[8].get_weights()))

        new_train_part1_tc = self.__network_TC.predict(train_x_tc)
        new_test_part1_tc = self.__network_TC.predict(test_x_tc)



        # print(np.array(new_train_from_prediction))
        # print("input_shape",train_x_tc.shape)
        # print("new_train_from_prediction shape", np.array(new_train_part1_tc).shape)

        ##Second MOdel
        train_x_ec, input_shape = self.__reshape_for_cnn(trainxEC)
        test_x_ec, _ = self.__reshape_for_cnn(test_x_EC)

        model_EC = Sequential()
        model_EC.add(Conv2D(filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu", input_shape=input_shape))  # 1.2 know terms of entities importancy #0
        model_EC.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 5)))  # 2. get max important term five by five #1
        model_EC.add(Conv2D(filters=64, kernel_size=(14, 4), strides=1, padding="same", activation="relu", input_shape=input_shape))  # 3 know entities importancy #2
        model_EC.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4)))  # 4. get entity iportancy by query phrace #3
        model_EC.add(AveragePooling2D(pool_size=(14, 1), strides=(14, 1)))  # 5 average of entity iportancy on total query #4
        # model_EC.add(Conv2D(filters=256, kernel_size=(5, 5), strides=5, padding="same",activation="relu"))  # 3 feature reduction #5
        model_EC.add(Flatten())  # 6
        # model_EC.add(Dense(100))  # 7
        # model_EC.add(Activation('relu'))  # 8
        model_EC.add(Dense(1, activation="linear"))  # 9

        ec_learning_rate = 0.0001
        adam = optimizers.Adam(lr=ec_learning_rate)
        model_EC.compile(optimizer=adam, loss="mse", metrics=["accuracy"])

        print("\n\nModel Entity Centric Fitting")
        model_EC.fit(train_x_ec, train_y, validation_data=(test_x_ec, test_y), epochs=two_inner_models_epochs, batch_size=128, verbose=2)

        # print(model_EC.summary())
        print("\n\nModel Entity Centric Fitted")



        self.__network_EC = Sequential()
        self.__network_EC.add(Conv2D(weights=model_EC.layers[0].get_weights(), filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu", input_shape=input_shape))
        self.__network_EC.add(MaxPooling2D(weights=model_EC.layers[1].get_weights(), pool_size=(1, 5), strides=(1, 5)))  # 2. get max important term five by five #1
        self.__network_EC.add(Conv2D(weights=model_EC.layers[2].get_weights(), filters=64, kernel_size=(14, 4), strides=1, padding="same", activation="relu", input_shape=input_shape))  # 3 know entities importancy #2
        self.__network_EC.add(MaxPooling2D(weights=model_EC.layers[3].get_weights(), pool_size=(1, 4), strides=(1, 4)))  # 4. get entity iportancy by query phrace #3
        self.__network_EC.add(AveragePooling2D(weights=model_EC.layers[4].get_weights(), pool_size=(14, 1), strides=(14, 1)))  # 5 average of entity iportancy on total query #4
        # self.__network_EC.add(Conv2D(weights=model_EC.layers[5].get_weights(), filters=256, kernel_size=(5, 5), strides=5, padding="same",activation="relu"))  # 3 feature reduction #5
        # self.__network_EC.add(Flatten(weights=model_EC.layers[6].get_weights()))  # 6

        self.__network_EC.add(Flatten(weights=model_EC.layers[5].get_weights()))  # 6


        # self.__network_EC.add(Dense(100, weights=model_EC.layers[7].get_weights()))  # 7
        # self.__network_EC.add(Activation('relu', weights=model_EC.layers[8].get_weights()))  # 8

        new_train_part2_ec = self.__network_EC.predict(train_x_ec)
        new_test_part2_ec = self.__network_EC.predict(test_x_ec)
        # print("input_shape", train_x_ec.shape)
        # print("new_train_part2_ec shape", np.array(new_train_part2_ec).shape)

        #create new train set from pediction of two models
        new_train_X_for_model3 = []
        for instance_tc, instance_ec in zip(new_train_part1_tc.tolist(), new_train_part2_ec.tolist()):
            new_instance = instance_tc + instance_ec
            # new_instance = np.array([instance_tc , instance_ec]).mean(axis=0)
            new_train_X_for_model3.append(new_instance)

        new_test_X_for_model3 = []
        for instance_test_tc, instance_ec_test in zip(new_test_part1_tc.tolist(), new_test_part2_ec.tolist()):
            new_test_instance = instance_test_tc + instance_ec_test
            # new_test_instance = np.array([instance_test_tc , instance_ec_test]).mean(axis=0)
            new_test_X_for_model3.append(new_test_instance)

        new_train_X_for_model3 = np.array(new_train_X_for_model3)
        new_train_Y_for_model3 = train_y
        new_test_X_for_model3 = np.array(new_test_X_for_model3)
        new_test_Y_for_model3 = test_y

        print(new_train_X_for_model3.shape)

        train_x = new_train_X_for_model3
        test_x  = new_test_X_for_model3
        test_y = new_test_Y_for_model3

        self.__network = Sequential()

        ############CNN ADDED#############
        # train_x = np.expand_dims(train_x, axis=2)
        # test_x = np.expand_dims(test_x, axis=2)
        # # self.__network.add(Conv1D(filters= 128, kernel_size=5, input_shape = (200, 1)))
        # self.__network.add(Conv1D(filters= 128, kernel_size=5, input_shape = (300, 1)))
        # self.__network.add(Flatten())
        ############CNN ADDED#############


        ##########layers added static !
        for i in range(0, len(self.__layers)):
            self.__network.add(Dense(self.__layers[i]))
            if self.__activation[i] == "LeakyReLU":
                self.__network.add(LeakyReLU(alpha=0.2))
            else:
                self.__network.add(Activation(self.__activation[i]))
            self.__network.add(Dropout(self.__dropout))
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


        print("\n\nModel Two Step Merge Fitting")
        print(self.__network.summary())

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

        print("\n\nModel Two Step Merge Fitted :)")
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

        new_test_part1_tc = self.__network_TC.predict(test_x_tc)
        new_test_part2_ec = self.__network_EC.predict(test_x_ec)
        new_test_X_for_model3 = []
        for instance_test_tc, instance_ec_test in zip(new_test_part1_tc.tolist(), new_test_part2_ec.tolist()):
            new_test_instance = instance_test_tc + instance_ec_test
            # new_test_instance = np.array([instance_test_tc , instance_ec_test]).mean(axis=0)
            new_test_X_for_model3.append(new_test_instance)

        new_test_X_for_model3 = np.array(new_test_X_for_model3)
        test_x = new_test_X_for_model3

        #################CNN ADDED ########################
        # test_x = np.expand_dims(test_x, axis=2)
        #################CNN ADDED ########################




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