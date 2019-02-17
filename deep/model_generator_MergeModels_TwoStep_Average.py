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
        f = open("tahlil_khoroji_2model_tc_ec_vase_mergeshun.txt","a")
        print('test_y.shape', test_y.shape)
        """ Performs training on train_x instances"""
        # train_x = np.array(train_x)
        # print(train_x.shape)
        # test_x = np.array(test_x)
        #First model Static
        # print(train_xTC.shape)
        # print(trainxEC.shape)

        two_inner_models_epochs = 100
        inner_models_verbose = 0
        self.__verbose = 2

        train_x_tc, input_shape = self.__reshape_for_cnn(train_xTC)
        test_x_tc , _ = self.__reshape_for_cnn(test_x_TC)
        # train_x_tc, input_shape = self.__reshape_for_cnn(np.array(train_x[0]))
        # test_x_tc , _ = self.__reshape_for_cnn(np.array(test_x[0]))

        self.__network_TC = Sequential()
        self.__network_TC.add(Conv2D(filters=64, kernel_size= (5,5),strides=(1,1), padding="same", activation="relu", input_shape=input_shape))
        self.__network_TC.add(MaxPooling2D())

        self.__network_TC.add(Conv2D(filters=16, kernel_size= (10,10),strides=(1,1), padding="same", activation="relu"))
        self.__network_TC.add(MaxPooling2D())

        self.__network_TC.add(Conv2D(filters=256, kernel_size= (32,32),strides=(1,1), padding="same", activation="relu"))
        self.__network_TC.add(MaxPooling2D())

        self.__network_TC.add(Flatten())

        self.__network_TC.add(Dense(100))
        self.__network_TC.add(Activation('relu'))

        self.__network_TC.add(Dense(1))
        self.__network_TC.add(Activation('linear'))

        tc_learning_rate = 0.0001
        adam = optimizers.Adam(lr=tc_learning_rate)
        self.__network_TC.compile(optimizer=adam, loss="mse", metrics=["accuracy"])

        print("\n\nModel Type Centric Fitting")

        #two_inner_models_epochs
        self.__history = self.__network_TC.fit(train_x_tc, train_y, validation_data=(test_x_tc, test_y), epochs=two_inner_models_epochs,
                           batch_size=128, verbose=inner_models_verbose)

        # print(model_TC.summary())
        print("\n\nModel Type Centric Fitted")

        # print(np.array(model_TC.layers[8].get_weights()[0]).shape)
        # print(np.array(model_TC.layers[8].get_weights()[1]).shape)




        # print(np.array(new_train_from_prediction))
        # print("input_shape",train_x_tc.shape)
        # print("new_train_from_prediction shape", np.array(new_train_part1_tc).shape)

        ##Second MOdel
        train_x_ec, input_shape = self.__reshape_for_cnn(trainxEC)
        test_x_ec, _ = self.__reshape_for_cnn(test_x_EC)

        self.__network_EC = Sequential()
        self.__network_EC.add(Conv2D(filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu", input_shape=input_shape))  # 1.2 know terms of entities importancy #0
        self.__network_EC.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 5)))  # 2. get max important term five by five #1
        self.__network_EC.add(Conv2D(filters=64, kernel_size=(14, 4), strides=1, padding="same", activation="relu", input_shape=input_shape))  # 3 know entities importancy #2
        self.__network_EC.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4)))  # 4. get entity iportancy by query phrace #3
        self.__network_EC.add(AveragePooling2D(pool_size=(14, 1), strides=(14, 1)))  # 5 average of entity iportancy on total query #4
        self.__network_EC.add(Flatten())  # 6
        self.__network_EC.add(Activation('relu'))  # 8
        self.__network_EC.add(Dense(1, activation="linear"))  # 9

        ec_learning_rate = 0.0001
        adam = optimizers.Adam(lr=ec_learning_rate)
        self.__network_EC.compile(optimizer=adam, loss="mse", metrics=["accuracy"])

        print("\n\nModel Entity Centric Fitting")
        self.__network_EC.fit(train_x_ec, train_y, validation_data=(test_x_ec, test_y), epochs=two_inner_models_epochs, batch_size=128, verbose=inner_models_verbose)

        # print(model_EC.summary())
        print("\n\nModel Entity Centric Fitted")
        self.__network = self.__network_TC
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
        predict_values = []
        test_x_tc , _ = self.__reshape_for_cnn(test_x_TC)
        test_x_ec, _ = self.__reshape_for_cnn(test_x_EC)

        ###########
        # test_x_ec = np.array(test_x[1])
        ###########


        test_x = [test_x_tc, test_x_ec]

        new_test_part1_tc = self.__network_TC.predict(test_x_tc)
        new_test_part2_ec = self.__network_EC.predict(test_x_ec)
        new_test_X_for_model3 = []


        cnt = 0
        for instance_test_tc, instance_ec_test in zip(new_test_part1_tc.tolist(), new_test_part2_ec.tolist()):
            instance_test_tc = instance_test_tc[0]
            instance_ec_test = instance_ec_test[0]
            if not test_x_ec[cnt].any():
                # instance_ec_test = instance_test_tc[:len(instance_ec_test)] #az avale instance tc slice bardar !
                # instance_ec_test = np.zeros(len(instance_ec_test)).tolist()
                predict = float(instance_test_tc)
                predict_values.append(predict)

            else:
                predict = (float(instance_test_tc) + float(instance_ec_test))/2
                predict_values.append(predict)

            cnt+=1



        new_test_X_for_model3 = np.array(new_test_X_for_model3)
        test_x = new_test_X_for_model3

        #################CNN ADDED ########################
        # test_x = np.expand_dims(test_x, axis=2)
        #################CNN ADDED ########################




        # test_x , _ = self.__reshape_for_cnn(test_x)
        result = dict()

        output_activation = self.__activation[len(self.__activation)-1]

        if  output_activation == "linear":
            # predict_values = self.__network.predict(test_x)
            result["predict"] = predict_values

        elif output_activation == "softmax":
            predict_values = self.__network.predict_classes(test_x)
            predict_prob_values = self.__network.predict_proba(test_x)
            result["predict"] = predict_values
            result["predict_prob"] = predict_prob_values

        if test_y is not None:
            result["loss_mean"], result["acc_mean"] = 0,0

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