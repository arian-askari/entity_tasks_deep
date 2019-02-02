import os, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger

class Model_Generator():
    def __init__(self, layers, activation=[], optimizer="adam", loss="mse", learning_rate=0.0001,
                 lambd=0, batch_size=100, epochs=1000, dropout=0.0, verbose=2, category="regression"):
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
        self.__model_name = self.model_name()
        self.__network = None
        self.__history = None
        self.__csv_log_path = ""
        self.__sess = None
        print("layers =", layers, ", activation =", activation, ", optimizer =", optimizer, ", loss =", loss,
              ", lambda =", lambd, ", learning_rate =", learning_rate, ", batch_size =", batch_size, " ,epochs =", epochs,
              " ,dropout =", dropout, " ,verbose =", verbose, " ,category =", category)

    def set_csv_log_path(self, csv_log_path):
        self.__csv_log_path = csv_log_path

    def fit(self, train_x, train_y, input_dim): #input_dim example: (600,)
        """ Performs training on train_x instances"""

        self.__network = Sequential()
        self.__network.add(Dense(self.__layers[0], input_shape=input_dim))
        self.__network.add(Activation(self.__activation[0]))
        self.__network.add(Dropout(self.__dropout))

        for i in range(1, len(self.__layers)):
            self.__network.add(Dense(self.__layers[i]))
            self.__network.add(Activation(self.__activation[i]))
            self.__network.add(Dropout(self.__dropout))


        if (self.__optimizer == "adam"):
            adam = optimizers.Adam(lr=self.__learning_rate)
            self.__network.compile(optimizer=adam, loss=self.__loss, metrics=["accuracy"])

        self.__network.compile(optimizer=self.__optimizer, loss=self.__loss, metrics=["accuracy"])

        if len(self.__csv_log_path) > 0:
            csv_logger = CSVLogger(self.__csv_log_path, append=False, separator=',')
            self.__history = self.__network.fit(train_x, train_y, epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose, callbacks=[csv_logger])
        else:
            self.__history = self.__network.fit(train_x, train_y, epochs=self.__epochs, batch_size=self.__batch_size, verbose=self.__verbose)

        result = dict()
        result["model"] = self.__network
        result["train_loss_latest"] = self.__history.history['loss'][-1]
        result["train_acc_mean"] = np.mean(self.__history.history['acc'])

        result["train_loss"] = self.__history.history['loss']
        result["train_acc"] = self.__history.history['acc']

        print("\nmodel summary:\n--------------")
        print(self.__network.summary())

        return result

    def get_train_loss_mean(self):
        """ get loss and acc during training."""
        if self.__network is not None:
            result = dict()
            result["train_loss_latest"] = self.__history.history['loss'][-1]
            result["train_acc_mean"] = np.mean(self.__history.history['acc'])

            result["train_loss"] = self.__history.history['loss']
            result["train_acc"] = self.__history.history['acc']
            return self.__network
        else:
            return None

    def get_model_obj(self):
        return self.__network


    def predict(self, test_x, test_y=None):
        """ Performs prediction."""
        result = dict()

        output_activation = self.__activation[len(self.__activation)-1]

        if  output_activation == "linear":
            predict_values = self.__network.predict(test_x)
            result["predict"] = predict_values

        elif output_activation == "linear":
            predict_values = self.__network.predict_classes(test_x)
            predict_prob_values = self.__network.predict_proba(test_x)
            result["predict"] = predict_values
            result["predict_prob"] = predict_prob_values

        if test_y is not None:
            test_y = np.array(test_y)
            print(test_y)
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
        model_name += "_" + str(self.__category)
        return model_name

    def get_model_name(self):
        return self.__model_name


def example():
    layers = [100, 100, 1]  # regression sample
    # layers = [100, 100, 2]  # classification sample

    activation =  ["relu", "relu", "linear"]
    # activation =  ["relu", "relu", "softmax"]

    X_Train = np.array([[1, 2, 3], [100, 300, 500]])

    Y_Train = np.array([0, 1])  # regression sample
    # Y_Train = np.array([[1, 0], [0, 1]])  # classification sample

    X_Test = np.array([ [2, 4, 6], [200, 600, 1000]])

    Y_Test = np.array([0, 1])  # regression sample
    # Y_Test = np.array([[1, 0], [0, 1]])  # classification sample

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