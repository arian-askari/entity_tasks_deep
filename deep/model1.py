import os, subprocess, json, ast, sys, re, random

import seaborn as sns
import numpy as np

import keras_metrics
from keras.layers.core import Dense, Activation
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.layers import *
from keras.models import *

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models.keyedvectors import KeyedVectors

from elasticsearch import Elasticsearch

ANALYZER_STOP = "stop_en"
es = Elasticsearch(timeout=10000)

dirname = os.path.dirname(__file__)

# queries_path = os.path.join(dirname, '../data/dbpedia-v2/queries_stopped_with_type.json')
queries_path = os.path.join(dirname, '../data/dbpedia-v1/queries_type_retrieval.json')
# queries_path = os.path.join(dirname, '../data/dbpedia-v2/queries_stopped.json')
qrel_types_path = os.path.join(dirname, '../data/types/qrels-tti-dbpedia.txt')
qrels_types_dict = dict()

word2vec_train_set_path = '/home/arian/workSpaces/entityArticle/entity-attr-resources/mycode/data/GoogleNews-vectors-negative300.bin'
# word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True, limit=100000)
# word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True)
word_vectors = []

import csv
with open(qrel_types_path) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"): # can also use delimiter="\t" rather than giving a dialect.
        query_id = line[0]
        query_target_type = line[1]
        rel_level = line[2]
        # print(line)
        if query_id not in qrels_types_dict:
            qrels_types_dict[query_id] = [(query_target_type, rel_level)]
        elif query_id in qrels_types_dict:
            qrels_types_dict[query_id].append((query_target_type, rel_level))

def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))

def getTokens(text, index):
    query_tokenied_list = es.indices.analyze(index=index, params={"text": text, "analyzer": ANALYZER_STOP})
    tokens = []
    for t in sorted(query_tokenied_list["tokens"], key=lambda x: x["position"]):
        tokens.append(t["token"])
    return tokens

def getVector(word):
    if word in word_vectors:
        vector = word_vectors[word]
        return vector
    else:
        print(word+ "\t not in w2v google news vocab !")
        return [];

def get_average_w2v(sentence):#bayad vorodish token bashe ! chon khdoe tolid token vase queries behtare split by space bashe
    #ama vase text e type shayad behtar bashe az rooye index kolan term vector begiram token haro masalan!

    #bayaad ba elastic inja ba term joda konam !
    #getTokens(text, index)
    #ama felan ba space split mikonam ke codo test konam o sade tar bashe !

    tokens = sentence.split(' ')
    first_token = tokens[0]
    # print('first_token ', first_token)
    np_array = None
    vector = getVector(first_token)
    
    if len(vector)>0:
        np_array = np.array([np.array(vector)])

    for token in tokens[1:]:
        print('token ', token)
        vector = getVector(token)
        if len(vector)>0:
            tmp_array = np.array(vector)
            np_array = np.concatenate((np_array, [tmp_array]))
    
    # print(np_array)
    # print('\n\n-----\n\n')
    vector_mn = np_array.mean(axis=1)     # to take the mean of each row, 300D vec
    return vector_mn

def model_type_retrieval_v1(train_X, train_Y, test_X, test_Y):
    avg_q_ = [] #baes bani error e dar nahayat!

    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(20, input_shape=(avg_q_,)))

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
    model_type_retrieva_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])

    train_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
    train_Y = [1, 1, '...', 0]

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1)

    test_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
    test_Y = [1, 1, '...', 0]

    print(model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=0))

    loss, accuracy, precision, recall = model_type_retrieva_v1.evaluate(test_X, test_Y, verbose=0)
    print("Accuracy = {:.2f}".format(accuracy))
    print("precision = {:.2f}".format(accuracy))
    print("recall = {:.2f}".format(accuracy))
    print("loss = {:.2f}".format(loss))




'''
# Training Phase, AND, #Test Phase !
'''
# with open(queries_path, 'r') as ff:
#     # print(q)
#     # avg_q_ = get_average_w2v(q)
#     # print(avg_q_)
#
#     queries_json = json.load(ff)
#
#     queries_dict = dict(queries_json)
#
#     queries_for_select_test_set = dict(queries_dict)
#     k_fold = 5
#     for i in range(k_fold):
#         fold_size = int(len(queries_dict)/k_fold)  #168/6=28
#
#         queries_for_test_set = None
#         if (i + 1) == k_fold:
#             queries_for_test_set = list(queries_for_select_test_set.items())
#         else:
#             queries_for_test_set = random.sample(list(queries_for_select_test_set.items()), fold_size)
#
#         queries_for_select_test_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)
#
#         queries_for_train = substrac_dicts(queries_dict, queries_for_test_set)