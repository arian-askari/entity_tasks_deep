"""
Creates Feed forward Neural nets

Author: Faegheh Hasibi
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Feedforward(object):
    def __init__(self, layers, activation="relu", optimizer="adam", learning_rate="0.001",
                 lambd=0, batch_size=100, epochs=1000, category="regression"):
        self.__layers = layers
        self.__activation = activation
        self.__optimizer = optimizer
        self.__learning_rate = learning_rate
        self.__lambd = lambd  # for L2 regularization
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__category = category
        self.__x = tf.placeholder(tf.float32, [None, self.__layers[0]], name="X")
        self.__y = tf.placeholder(tf.float32, [None, self.__layers[-1]], name="y")
        self.__network = None
        self.__sess = None
        print("layers =", layers, ", activation =", activation, ", optimizer =", optimizer, ", lambda =", lambd,
              ", learning_rate =", learning_rate, ", batch_size =", batch_size, " ,epochs =", epochs)

    def get_activation(self, layer_number):
        """Returns the activation function."""
        # last layer in regression
        if (self.__category == "regression") and (layer_number == len(self.__layers) - 2):
            return tf.identity
        elif self.__activation == "relu":
            return tf.nn.relu
        elif self.__activation == "softmax":
            return tf.nn.softmax
        else:
            raise Exception("Activation function is not valid!")

    def init_coefs(self):
        """initializes weights and biases of the model.
        Use the initialization method recommended by Glorot et al.
        """
        weights, biases = [], []
        for i in range(len(self.__layers)-1):
            init_bound = np.sqrt(6. / (self.__layers[i] + self.__layers[i+1]))
            weights.append(tf.Variable(tf.random_uniform([self.__layers[i], self.__layers[i+1]],
                                                         minval=-init_bound, maxval=init_bound), name="w"+str(i)))
            biases.append(tf.Variable(tf.random_uniform([self.__layers[i+1]],
                                                        minval=-init_bound, maxval=init_bound), name="b"+str(i)))
        return weights, biases

    def gen_network(self, input_layer, weights, biases):
        """Builds the network.
        Formulation: z^{i+1} = f(z^i w^i + b^i)
        """
        z = [input_layer]
        for i in range(len(self.__layers)-1):
            f = self.get_activation(i)
            z.append(f(tf.matmul(z[i], weights[i]) + biases[i]))
        return z[-1]

    def gen_regularizer(self, weights):
        """Generates L2 regularizer."""
        l2_reg = tf.Variable(initial_value=0.0)
        for w in weights:
            l2_reg = l2_reg + tf.nn.l2_loss(w)
        l2_reg = self.__lambd * l2_reg
        return l2_reg

    def gen_batches(self, n):
        """Generator to create slices containing batch_size elements, from 0 to n.

        The last slice may contain less than batch_size elements, when batch_size
        does not divide n.
        """
        start = 0
        for _ in range(int(n // self.__batch_size)):
            end = start + self.__batch_size
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)

    def fit(self, train_x, train_y, val_x=None, val_y=None):
        """Trains the neural net."""
        train_x, train_y = train_x.as_matrix(), train_y.as_matrix()
        # Cost function
        weights, biases = self.init_coefs()
        self.__network = self.gen_network(self.__x, weights, biases)
        cost = tf.reduce_mean(tf.square(self.__network - self.__y)) + self.gen_regularizer(weights)
        train = tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(cost)

        # training
        self.__sess = tf.InteractiveSession()  # config=tf.ConfigProto(log_device_placement=True))
        tf.global_variables_initializer().run()
        for epoch in range(self.__epochs):
            train_x, train_y = shuffle(train_x, train_y)
            for batch_slice in self.gen_batches(train_x.shape[0]):
                batch_xs = train_x[batch_slice]
                batch_ys = train_y[batch_slice]
                _, loss = self.__sess.run([train, cost], feed_dict={self.__x: batch_xs, self.__y: batch_ys})
            train_pred = self.predict(train_x)
            val_str = ""
            if (val_x is not None) and (val_y is not None):
                val_pred = self.predict(val_x)
                val_str = ", r2_val = " + str(np.round(metrics.r2_score(val_y, val_pred, multioutput="raw_values"), 4)) +\
                    ", mse_val = " + str(np.round(metrics.mean_squared_error(val_y, val_pred, multioutput="raw_values"), 4))
            print("EPOCH", epoch, ": loss =", round(loss, 4), ", mse =",
                  np.round(metrics.mean_squared_error(train_y, train_pred, multioutput="raw_values"), 4), val_str)

    def predict(self, test_x, batch_computation=False):
        """ Performs prediction.

        :param test_x: Test data, numpy array or dataframe
        :param batch_computation: if True, computes the network in batches and then concatenate the results
        :return: predicted results
        """
        test_x = test_x.as_matrix() if type(test_x) != np.ndarray else test_x
        if batch_computation:
            results = []
            for batch_slice in self.gen_batches(test_x.shape[0]):
                batch_xs = test_x[batch_slice]
                results.append(self.__sess.run(self.__network, feed_dict={self.__x: batch_xs}))
            return np.concatenate(results, axis=0)
        else:
            return self.__sess.run(self.__network, feed_dict={self.__x: test_x})

