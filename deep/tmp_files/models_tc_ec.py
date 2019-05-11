import os, subprocess, json, ast, sys, re, random
import pandas as pd
import csv

import numpy as np

import keras
import keras_metrics
from keras.layers.core import Dense, Activation
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras import optimizers
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as k
import deep.train_set_generator as tsg
from collections import Counter

import math
from elasticsearch import Elasticsearch

ANALYZER_STOP = "stop_en"
es = Elasticsearch(timeout=10000)

dirname = os.path.dirname(__file__)

# queries_path = os.path.join(dirname, '../data/types/sig17/queries-v1.json')
queries_path = os.path.join(dirname, '../data/dbpedia-v1/queries_type_retrieval.json')


# queries_path = os.path.join(dirname, '../data/dbpedia-v2/queries_stopped_with_type.json')
# queries_path = os.path.join(dirname, '../data/dbpedia-v1/queries_type_retrieval.json')
# queries_path = os.path.join(dirname, '../data/dbpedia-v2/queries_stopped.json')
qrel_types_path = os.path.join(dirname, '../data/types/qrels-tti-dbpedia.txt')
qrels_types_dict = dict()

word2vec_train_set_path = '/home/arian/workSpaces/entityArticle/entity-attr-resources/mycode/data/GoogleNews-vectors-negative300.bin'
# word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True, limit=100000)
# word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True)
word_vectors = []

# models_path = os.path.join(dirname, '../data/runs/sig17/model_v')


# models_path = os.path.join(dirname, '../data/runs/translation/model_merged_v')
models_path = os.path.join(dirname, '../data/runs/translation/sig17/merged_model_ec_tcNew')

# models_path = os.path.join(dirname, '../data/runs/translation/model_v')

def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))



def model_type_retrieval_v4(train_X, train_Y, test_X, test_Y, train_X2, test_X2): #one_layer, count of neuron is count of types!
    # train_Y = np.argmax(train_Y, axis=1)

    # print(train_X2[0])
    img_rows = 14
    img_cols = 100
    train_X2 = np.array(train_X2)
    test_X2 = np.array(test_X2)
    input_shape = None
    if k.image_data_format() == 'channels_first':
        train_X2 = train_X2.reshape(train_X2.shape[0], 1, img_rows, img_cols)
        test_X2 = test_X2.reshape(test_X2.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        train_X2 = train_X2.reshape(train_X2.shape[0], img_rows, img_cols, 1)
        test_X2 = test_X2.reshape(test_X2.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)# print(test_X2[0])



    #more reshapingprint(type(test_X2))
    train_X2 = train_X2.astype('float32')
    test_X2 = test_X2.astype('float32')


    ##############

    img_rows = 14
    img_cols = 5
    train_X = np.array(train_X)
    test_X = np.array(test_X)
    input_shape2 = None
    if k.image_data_format() == 'channels_first':
        train_X = train_X.reshape(train_X.shape[0], 1, img_rows, img_cols)
        test_X = test_X.reshape(test_X2.shape[0], 1, img_rows, img_cols)
        input_shape2 = (1, img_rows, img_cols)
    else:
        train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
        test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)
        input_shape2 = (img_rows, img_cols, 1)# print(test_X2[0])


    ############################################################ faghat ec, sohbat ba reza !######################################################################
    # model_ec = Sequential()
    # model_ec.add(Conv2D(filters=5, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu", input_shape=input_shape))
    # model_ec.add(AveragePooling2D())
    # model_ec.add(Dropout(0.5))
    #
    #
    # model_ec.add(Flatten())
    #
    # model_ec.add(Dense(1000))
    # model_ec.add(Activation('relu'))
    #
    # model_ec.add(Dense(100))
    # model_ec.add(Activation('relu'))
    #
    # model_ec.add(Dense(8))
    #
    # model_ec.add(Activation('softmax'))
    #
    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    #
    # model_ec.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # model_ec.fit(train_X2, train_Y, epochs=200, batch_size=100, verbose=2)
    #
    # # predict_classes = model_ec.predict(test_X2, batch_size=100)
    # predict_classes = model_ec.predict(test_X2)
    # predicted_prob = model_ec.predict_proba(test_X2)



    ##################NDCG 5, 40% hododan vali faghat ye fold :D ! in config ! ##########################################
    # model_ec = Sequential()
    # model_ec.add(Conv2D(filters=5, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu",
    #                     input_shape=input_shape))
    # model_ec.add(AveragePooling2D())
    # model_ec.add(Dropout(0.5))
    #
    # model_ec.add(Flatten())
    #
    # model_ec.add(Dense(1000))
    # model_ec.add(Activation('relu'))
    #
    # model_ec.add(Dense(100))
    # model_ec.add(Activation('relu'))
    #
    #
    # model_tc = Sequential()
    # model_tc.add(Dense(1000, input_shape=(600,)))
    # model_tc.add(Activation('relu'))
    # model_tc.add(Dropout(0.5))
    #
    # model_tc.add(Dense(100, input_shape=(600,)))
    # model_tc.add(Activation('relu'))
    # model_tc.add(Dropout(0.5))
    #
    # merged_tc_ec = Add()([model_tc.output, model_ec.output])
    # merged_tc_ec = Dense(8)(merged_tc_ec)
    # merged_tc_ec = Activation('softmax')(merged_tc_ec)
    #
    # merged_q1_d1_newModel = Sequential()
    # merged_q1_d1_newModel = Model([model_tc.input, model_ec.input], merged_tc_ec)
    #
    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # merged_q1_d1_newModel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # merged_q1_d1_newModel.fit([train_X, train_X2], train_Y, epochs=100, batch_size=100, verbose = 2)
    #
    # predict_classes = merged_q1_d1_newModel.predict([test_X,test_X2])
    # predicted_prob = merged_q1_d1_newModel.predict([test_X,test_X2])
    ##################NDCG 5, 40% hododan ! in config ! ##########################################


    ##################NDCG 5, 37%  rooye hame fold ha ##########################################
    # model_ec = Sequential()
    # model_ec.add(Conv2D(filters=5, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu",
    #                     input_shape=input_shape))
    # model_ec.add(AveragePooling2D())
    # model_ec.add(Dropout(0.5))
    #
    # model_ec.add(Flatten())
    #
    # model_ec.add(Dense(100))
    # model_ec.add(Activation('relu'))
    # model_ec.add(Dropout(0.5))
    #
    # model_tc = Sequential()
    # model_tc.add(Dense(100, input_shape=(600,)))
    # model_tc.add(Activation('relu'))
    # model_tc.add(Dropout(0.5))
    #
    #
    # merged_tc_ec = Add()([model_tc.output, model_ec.output])
    # merged_tc_ec = Dense(8)(merged_tc_ec)
    # merged_tc_ec = Activation('softmax')(merged_tc_ec)
    #
    # merged_q1_d1_newModel = Sequential()
    # merged_q1_d1_newModel = Model([model_tc.input, model_ec.input], merged_tc_ec)
    #
    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # merged_q1_d1_newModel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # merged_q1_d1_newModel.fit([train_X, train_X2], train_Y, epochs=50, batch_size=100, verbose=2)
    #
    # predict_classes = merged_q1_d1_newModel.predict([test_X, test_X2])
    # predicted_prob = merged_q1_d1_newModel.predict([test_X, test_X2])
    ##################NDCG 5, 37%  rooye hame fold ha ##########################################


    ##################NDCG 5, 40%  rooye hame fold ha ##########################################
    # model_ec = Sequential()
    # model_ec.add(Conv2D(filters=5, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu",
    #                     input_shape=input_shape))
    # model_ec.add(MaxPooling2D())
    # model_ec.add(Dropout(0.5))
    #
    # model_ec.add(Flatten())
    #
    # model_ec.add(Dense(100))
    # model_ec.add(Activation('relu'))
    # model_ec.add(Dropout(0.5))
    #
    # model_tc = Sequential()
    # model_tc.add(Dense(100, input_shape=(600,)))
    # model_tc.add(Activation('relu'))
    # model_tc.add(Dropout(0.5))
    #
    #
    # merged_tc_ec = Add()([model_tc.output, model_ec.output])
    # merged_tc_ec = Dense(8)(merged_tc_ec)
    # merged_tc_ec = Activation('softmax')(merged_tc_ec)
    #
    # merged_q1_d1_newModel = Sequential()
    # merged_q1_d1_newModel = Model([model_tc.input, model_ec.input], merged_tc_ec)
    #
    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # merged_q1_d1_newModel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # merged_q1_d1_newModel.fit([train_X, train_X2], train_Y, epochs=50, batch_size=100, verbose=2)
    #
    # predict_classes = merged_q1_d1_newModel.predict([test_X, test_X2])
    # predicted_prob = merged_q1_d1_newModel.predict([test_X, test_X2])
    ##################NDCG 5, 40%  rooye hame fold ha ##########################################


    ##last config
    # model_ec = Sequential()
    # model_ec.add(Conv2D(filters=16, kernel_size=(2, 2), padding="same", strides=(1, 1), activation="relu",
    #                     input_shape=input_shape))
    # model_ec.add(MaxPooling2D())
    # model_ec.add(Dropout(0.6))
    #
    # model_ec.add(Flatten())
    #
    # model_ec.add(Dense(100))
    # model_ec.add(Activation('relu'))
    # model_ec.add(Dropout(0.6))
    #
    # model_tc = Sequential()
    # model_tc.add(Dense(100, input_shape=(600,)))
    # model_tc.add(Activation('relu'))
    # model_tc.add(Dropout(0.6))
    #
    # merged_tc_ec = Add()([model_tc.output, model_ec.output])
    # merged_tc_ec = Dense(8)(merged_tc_ec)
    # merged_tc_ec = Activation('softmax')(merged_tc_ec)
    #
    # merged_q1_d1_newModel = Sequential()
    # merged_q1_d1_newModel = Model([model_tc.input, model_ec.input], merged_tc_ec)
    #
    # sgd = optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
    # merged_q1_d1_newModel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # merged_q1_d1_newModel.fit([train_X, train_X2], train_Y, epochs=50, batch_size=256, verbose=2)
    #
    # predict_classes = merged_q1_d1_newModel.predict([test_X, test_X2])
    # predicted_prob = merged_q1_d1_newModel.predict([test_X, test_X2])
    ##last config
    # model_ec = Sequential()
    # model_ec.add(Conv2D(filters=16, kernel_size=(2, 2), padding="same", strides=(1, 1), activation="relu",
    #                     input_shape=input_shape))
    # model_ec.add(MaxPooling2D())
    # model_ec.add(Dropout(0.6))
    #
    #
    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # model_ec.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # model_ec.fit([train_X, train_X2], train_Y, epochs=50, batch_size=256, verbose=2)
    #
    # predict_classes = model_ec.predict_classes([test_X, test_X2])
    # predicted_prob = model_ec.predicted_prob([test_X, test_X2])

    ###repeat ndcg 40, with top-k 5

    model_ec = Sequential()
    model_ec.add(Conv2D(filters=5, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu",
                        input_shape=input_shape))
    model_ec.add(MaxPooling2D())
    model_ec.add(Dropout(0.5))

    model_ec.add(Flatten())

    model_ec.add(Dense(100))
    model_ec.add(Activation('relu'))
    model_ec.add(Dropout(0.5))

    model_tc = Sequential()
    model_tc.add(Conv2D(filters=5, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu",
                        input_shape=input_shape2))
    model_tc.add(MaxPooling2D())
    model_tc.add(Dropout(0.5))

    model_tc.add(Flatten())

    model_tc.add(Dense(100))
    model_tc.add(Activation('relu'))
    model_tc.add(Dropout(0.5))


    merged_tc_ec = Add()([model_tc.output, model_ec.output])
    merged_tc_ec = Dense(8)(merged_tc_ec)
    merged_tc_ec = Activation('softmax')(merged_tc_ec)

    merged_q1_d1_newModel = Sequential()
    merged_q1_d1_newModel = Model([model_tc.input, model_ec.input], merged_tc_ec)

    sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    merged_q1_d1_newModel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    merged_q1_d1_newModel.fit([train_X, train_X2], train_Y, epochs=50, batch_size=100, verbose=2)

    predict_classes = merged_q1_d1_newModel.predict([test_X, test_X2])
    predicted_prob = merged_q1_d1_newModel.predict([test_X, test_X2])


    ##################################################################################################################################


    ##################################################################################################################################
    # input1 = keras.layers.Input(shape=(600,))
    # x1 = keras.layers.Dense(419, activation='relu')(input1)
    # x1 = keras.layers.Dropout(0.4)(x1)
    #
    # input2 = keras.layers.Input(shape=(1400,))
    # x2 = keras.layers.Dense(419, activation='relu')(input2)
    # x2 = keras.layers.Dropout(0.4)(x2)
    #
    # # equivalent to added = keras.layers.add([x1, x2])
    # added = keras.layers.Add()([x1, x2])
    #
    # # out = keras.layers.Dense(8, activation='softmax')(added)
    # out = keras.layers.Dense(1, activation='linear')(added)
    # model = keras.models.Model(inputs=[input1, input2], outputs=out)
    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # model.compile(optimizer=sgd, loss='mse', metrics=["accuracy"])
    #
    # model.fit([train_X, train_X2], train_Y, epochs=1000, batch_size=100, verbose=2)
    #
    # predict_classes = model.predict([test_X, test_X2])
    ##################################################################################################################################



    ##################################################################################################################################
    # model_ec = Sequential()
    # model_ec.add(Dense(419, input_shape=(1400,)))
    # model_ec.add(Activation('softmax'))
    #
    # model_tc = Sequential()
    # model_tc.add(Dense(419, input_shape=(600,)))
    # model_tc.add(Activation('softmax'))
    #
    # merged_q1_d1 = Sequential()
    #
    # merged_tc_ec = Add()([model_tc.output, model_ec.output])
    # merged_tc_ec = Dense(8)(merged_tc_ec)
    # merged_tc_ec = Activation('softmax')(merged_tc_ec)
    #
    # merged_q1_d1_newModel = Sequential()
    # merged_q1_d1_newModel = Model([model_tc.input, model_ec.input], merged_tc_ec)

    ##################################################################################################################################



    # optimizers.

    # merged_q1_d1_newModel.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    #
    # merged_q1_d1_newModel.fit([train_X,train_X2], train_Y, epochs=100, batch_size=1, verbose = 2)

        # predict_classes = merged_q1_d1_newModel.predict_classes([test_X,test_X2])
    # predicted_prob = merged_q1_d1_newModel.predict_proba([test_X,test_X2])


    # predicted_prob = model.predict([test_X,test_X2])
    # predict_classes = predicted_prob.argmax(axis=-1)
    predict_classes = predict_classes.tolist()
    print(predict_classes)
    print("\n\n\n\n\n")


    predict_classes = [c.index(max(c)) for c in predict_classes]

    print(predict_classes)

    return predict_classes, predicted_prob
    # return predict_classes

def model_type_retrieval_v3(train_X, train_Y, test_X, test_Y):
    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(1000, input_shape=(14, 100)))

    model_type_retrieva_v1.add(Dense(900))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(800))
    model_type_retrieva_v1.add(Activation('relu'))


    model_type_retrieva_v1.add(Dense(700))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(600))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(500))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(400))
    model_type_retrieva_v1.add(Activation('relu'))


    model_type_retrieva_v1.add(Dense(300))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(200))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(100))
    model_type_retrieva_v1.add(Activation('relu'))


    # model_type_retrieva_v1.add(Dense(50))
    # model_type_retrieva_v1.add(Activation('relu'))
    #
    # model_type_retrieva_v1.add(Dense(25))
    # model_type_retrieva_v1.add(Activation('relu'))
    #
    # model_type_retrieva_v1.add(Dense(12))
    # model_type_retrieva_v1.add(Activation('relu'))


    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))


    sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model_type_retrieva_v1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1, verbose = 2)

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)

    predict_classes = predict_classes.tolist()


    predict_classes = [predict_classes.index(max(c)) for c in  predict_classes]
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    # print(model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=2))
    # loss, accuracy = model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=2)
    # print("Accuracy = {:.2f}".format(accuracy))
    # print("loss = {:.2f}".format(loss))
    # print(predict_classes)
    # true_preditc_list = [np.where(r == 1)[0][0] for r in test_Y]
    # for true_predict, predicted in zip(true_preditc_list, predict_classes):
    #    print("true predict is : ", true_predict,"\t predit by model: ", predicted,"\n")

    return (predict_classes, predicted_prob)


def model_type_retrieval_v2(train_X, train_Y, test_X, test_Y):
    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(1000, input_shape=(14, 100)))

    #poooling ghaedtanaa bayad bezanam bad flatten konam!
    '''
    model.add(TimeDistributed(Dense(1)))
    model.add(AveragePooling1D())
    '''
    model_type_retrieva_v1.add(Flatten())


    model_type_retrieva_v1.add(Dense(800))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(600))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(400))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(200))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(100))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))


    sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model_type_retrieva_v1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=200, batch_size=1, verbose = 2)

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    # print(model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=2))
    # loss, accuracy = model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=2)
    # print("Accuracy = {:.2f}".format(accuracy))
    # print("loss = {:.2f}".format(loss))
    # print(predict_classes)
    # true_preditc_list = [np.where(r == 1)[0][0] for r in test_Y]
    # for true_predict, predicted in zip(true_preditc_list, predict_classes):
    #    print("true predict is : ", true_predict,"\t predit by model: ", predicted,"\n")

    return (predict_classes, predicted_prob)


def model_type_retrieval_v1(train_X, train_Y, test_X, test_Y):
    avg_q_ = [] #baes bani error e dar nahayat!

    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(20, input_shape=(14, 100)))

    model_type_retrieva_v1.add(Dense(37))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(59))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(56))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(14))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(3))
    model_type_retrieva_v1.add(Activation('relu'))

    # model_type_retrieva_v1.add(Dense(189))
    # model_type_retrieva_v1.add(Activation('relu'))

    # ouput layer (binary model!)
    # model_type_retrieva_v1.add(Dense(1))
    # model_type_retrieva_v1.add(Activation('sigmoid'))

    #outupt layer (classifaction model, for NDCG !)
    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))


    # model_type_retrieva_v1.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    # model_type_retrieva_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    model_type_retrieva_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    # train_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
    # train_Y = [1, 1, '...', 0]

    # model_type_retrieva_v1.fit(train_X, train_Y, epochs=20, batch_size=1)
    model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1, verbose = 0)

    # test_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
    # test_Y = [1, 1, '...', 0]

    # print(model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=0))
    # print(model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=0))

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    # loss, accuracy = model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=0)
    # print("Accuracy = {:.2f}".format(accuracy))
    # print("loss = {:.2f}".format(loss))
    # print(predicted_prob)
    # print(predict_classes)

    # true_preditc_list = [np.where(r == 1)[0][0] for r in test_Y]
    #for true_predict, predicted in zip(true_preditc_list, predict_classes):
    #    print("true predict is : ", true_predict,"\t predit by model: ", predicted,"\n")

    return (predict_classes, predicted_prob)


def get_trec_output(q_id_list, test_TYPES, test_Y, predict_classes, predict_probs):
    trec_output_str = ""
    trec_ouput_dict = dict()

    for q_id_test, q_candidate_type, true_predict, predict_class, predict_prob \
            in zip(q_id_list, test_TYPES, test_Y, predict_classes, predict_probs):
        # 1
        query_id = q_id_test

        # 2
        iter_str = "Q0"

        # 3
        doc_id = q_candidate_type

        # 4
        rank_str = "0"  # trec az in estefade nemikone, felan ino nemikhad dorost print konam:)

        # 5
        sim_score = (predict_class+1) * predict_prob[predict_class] # (predict_class+1), baraye inke baraye class 0, score e ehtemal sefr nashe, hamaro +1 kardam, dar kol tasiir nadare, vase class 7 ham 1 mishe va score ha relative mishan
        sim_score = str(sim_score)
        # sim_score = str(predict_class)

        # 6
        run_id = "Model_Deep"

        delimeter = "	"

        if query_id not in trec_ouput_dict:
            trec_ouput_dict[query_id] = [(doc_id, rank_str, sim_score, run_id)]
        else:
            trec_ouput_dict[query_id].append((doc_id, rank_str, sim_score, run_id))


    for query_id, detailsList in trec_ouput_dict.items():
        detailsList_sorted = sorted(detailsList, key = lambda x: x[2], reverse=True)
        i = 0
        for detail in detailsList_sorted:
            doc_id = detail[0]
            rank_str = str(i)
            sim_score = detail[2]
            run_id = detail[3]
            trec_output_str += query_id + delimeter + iter_str + delimeter + doc_id + delimeter + rank_str + delimeter + sim_score + delimeter + run_id + "\n"
            i += 1

    # trec_output_str += query_id + delimeter + iter_str + delimeter + doc_id + delimeter + rank_str + delimeter + sim_score + delimeter + run_id + "\n"
    return trec_output_str



'''
    {q_id: (q_body, q_type, q_type_rel_class)}
    raw_trainset_dict['q_id'][0]//get q_body of q_id !
    raw_trainset_dict['q_id'][2]//get q_type_rel_class of q_id !

# def model_type_retrieval_v1(train_X, train_Y, test_X, test_Y):
#
# train_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
# train_Y = [1, 1, '...', 0]
#
# model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1)
#
# test_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
# test_Y = [1, 1, '...', 0]
'''

def create_file(path, data):
    f = open(path, 'w')
    f.write(data)
    f.close()

def append_file(path, data):
    f = open(path, 'a')
    f.write(data)
    f.close()



trainset_translation_matrix = tsg.get_trainset_translation_matrix_average_w2v()
# trainset_translation_matrix = tsg.get_trainset_translation_matrix_score_e_average_w2v()
trainset_average_w2v = tsg.get_trainset_translation_matrix_type_tfidf_terms(k=5)
i = 0
with open(queries_path, 'r') as ff:
    # print(q)
    # avg_q_ = get_average_w2v(q)
    # print(avg_q_)
    trec_output_modelv1 = ""
    trec_output_modelv2 = ""
    trec_output_modelv3 = ""
    trec_output_modelv4 = ""

    modelv4_path = models_path + "4.run"
    create_file(modelv4_path, trec_output_modelv4)
    
    queries_json = json.load(ff)

    queries_dict = dict(queries_json)

    queries_for_select_test_set = dict(queries_dict)
    k_fold = 5
    for i in range(k_fold):
        q_id_train_list = []
        train_X = []
        train_X2 = []
        train_Y = []
        train_TYPES = []

        test_X = []
        test_X2 = []
        test_Y = []
        test_TYPES = []
        q_id_test_list = []



        fold_size = int(len(queries_dict)/k_fold)  #168/6=28

        queries_for_test_set = None
        if (i + 1) == k_fold:
            queries_for_test_set = list(queries_for_select_test_set.items())
        else:
            queries_for_test_set = random.sample(list(queries_for_select_test_set.items()), fold_size)

        queries_for_select_test_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)

        queries_for_train = substrac_dicts(queries_dict, queries_for_test_set)

        # train_set_average_dict[q_id] = [(merged_features, q_type_rel_class)]

        for query_ids_train in queries_for_train.keys():
            label_zero_count = 0
            q_id_train_set = trainset_average_w2v[query_ids_train]
            q_id_trani_set_translation = trainset_translation_matrix[query_ids_train]

            labels_count = Counter(elem[1] for elem in q_id_train_set)
            labels_more_than_zero = sum(1 for k, v in labels_count.items() if "0" not in k)
            labels_zero = labels_count["0"]
            # print(labels_count)
            # print(labels_more_than_zero)
            # print(labels_zero)

            repeat_count = math.ceil(labels_zero/labels_more_than_zero)

            intances_more_than_zero_X1 = [item for item in q_id_train_set if "0" not in item[1]]
            intances_more_than_zero_X2 = [item for item in q_id_trani_set_translation if "0" not in item[1]]

            for ins_x1, ins_x2 in zip(intances_more_than_zero_X1, intances_more_than_zero_X2):
                for l in range(repeat_count):
                    q_id_train_set.append(ins_x1)
                    q_id_trani_set_translation.append(ins_x2)

            shuffle_list =  list(zip(q_id_train_set, q_id_trani_set_translation))
            random.shuffle(shuffle_list)
            q_id_train_set, q_id_trani_set_translation = zip(*shuffle_list)

            arian = 0
            for train_set, train_set_translation in zip(q_id_train_set, q_id_trani_set_translation):
                if train_set[1]=="0":
                    label_zero_count += 1

                    # if (label_zero_count<=1): #important new commented !
                    train_X.append(train_set[0])

                    t = np.array(train_set_translation[0])
                    # t = t.flatten()     ####important, convert 14*100 to 1400 !
                    # t = t.reshape(14,100,1)
                    train_X2.append(t)
                    # train_X2.append(train_set_translation[0])

                    if train_set[1] != train_set_translation[1]:
                        print("bug !, two different train set record was merged!")
                        sys.exit(1)
                    train_Y.append(train_set[1])
                    train_TYPES.append(train_set[2])
                    q_id_train_list.append(query_ids_train)
                else:
                    train_X.append(train_set[0])

                    t = np.array(train_set_translation[0])
                    # t = t.flatten()     ####important, convert 14*100 to 1400 !
                    # t = t.reshape(14,100,1)
                    train_X2.append(t)

                    # train_X2.append(train_set_translation[0])
                    train_Y.append(train_set[1])
                    train_TYPES.append(train_set[2])
                    q_id_train_list.append(query_ids_train)

        train_Y = pd.get_dummies(train_Y)
        train_Y = train_Y.values.tolist()


        # train_X = np.array(train_X)
        train_Y = np.array(train_Y)

        for query_ids_test in queries_for_test_set:
            label_zero_count = 0
            q_id_test_set = trainset_average_w2v[query_ids_test[0]]
            q_id_test_set_translation = trainset_translation_matrix[query_ids_test[0]]
            for test_set, test_set_translation in zip(q_id_test_set,q_id_test_set_translation):
                if test_set[1]=="0":
                    label_zero_count += 1

                #if (label_zero_count<=1):
                test_X.append(test_set[0])

                t = np.array(test_set_translation[0])
                # t = t.flatten()     ####important, convert 14*100 to 1400 !
                # t = t.reshape(14,100,1)
                test_X2.append(t)
                # test_X2.append(test_set_translation[0])

                if (test_set_translation[1] != test_set[1]):
                    print("bug !, two different train set record was merged!")
                    sys.exit(1)
                test_Y.append(test_set[1])
                test_TYPES.append(test_set[2])
                q_id_test_list.append(query_ids_test[0])

        test_Y_one_hot = pd.get_dummies(test_Y)
        test_Y_one_hot = test_Y_one_hot.values.tolist()

        # test_X = np.array(test_X)
        test_Y_one_hot = np.array(test_Y_one_hot)

        ######################## generate trec output for IR measures :) ########################

        # predict_classes_v2, predicted_prob_v2 = model_type_retrieval_v2(train_X, train_Y, test_X, test_Y_one_hot)
        # trec_output_modelv2 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v2, predicted_prob_v2)
        #
        # predict_classes_v1, predicted_prob_v1 = model_type_retrieval_v1(train_X, train_Y, test_X, test_Y_one_hot)
        # trec_output_modelv1 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v1, predicted_prob_v1)
        #
        # predict_classes_v3, predicted_prob_v3 = model_type_retrieval_v3(train_X, train_Y, test_X, test_Y_one_hot)
        # trec_output_modelv3 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v3, predicted_prob_v3)
        #
        #
        predict_classes_v4, predicted_prob_v4 = model_type_retrieval_v4(train_X, train_Y, test_X, test_Y_one_hot, train_X2, test_X2)
        trec_output_modelv4 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v4, predicted_prob_v4)
        tmp = trec_output_modelv4.rstrip('\n')
        append_file(modelv4_path, trec_output_modelv4)
        trec_output_modelv4 = ""

        ######################## generate trec output for IR measures :) ########################
        print("\n-----------------------------------------------\n\n\n")
        

    # trec_output_modelv2 = trec_output_modelv2.rstrip('\n')
    # trec_output_modelv1 = trec_output_modelv1.rstrip('\n')
    # trec_output_modelv3 = trec_output_modelv3.rstrip('\n')

    # modelv2_path = models_path + "2.run"
    # modelv1_path = models_path + "1.run"
    # modelv3_path = models_path + "3.run"

    # create_file(modelv2_path, trec_output_modelv2)
    # create_file(modelv1_path, trec_output_modelv1)
    # create_file(modelv3_path, trec_output_modelv3)


    # sys.exit(1)