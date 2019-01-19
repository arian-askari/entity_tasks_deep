import os, subprocess, json, ast, sys, re, random
import pandas as pd
import csv
import seaborn as sns
import numpy as np

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

import deep.train_set_generator as tsg


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
models_path = os.path.join(dirname, '../data/runs/translation/model_v')

def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))



def model_type_retrieval_v4(train_X, train_Y, test_X, test_Y): #one_layer, count of neuron is count of types!
    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(419, input_shape=(14, 100)))

    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))


    sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # optimizers.
    model_type_retrieva_v1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1, verbose = 2)

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    return (predict_classes, predicted_prob)


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


def get_trec_output(q_id_list, test_TYPES, test_Y, predict_classes, predicted_prob):
    trec_output_str = ""
    trec_ouput_dict = dict()

    for q_id_test, q_candidate_type, true_predict, predict_class, predict_prob \
            in zip(q_id_list, test_TYPES, test_Y, predict_classes, predicted_prob):
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


trainset_average_w2v = tsg.get_trainset_translation_matrix_average_w2v()
i = 0
with open(queries_path, 'r') as ff:
    # print(q)
    # avg_q_ = get_average_w2v(q)
    # print(avg_q_)
    trec_output_modelv1 = ""
    trec_output_modelv2 = ""
    trec_output_modelv3 = ""
    trec_output_modelv4 = ""

    queries_json = json.load(ff)

    queries_dict = dict(queries_json)

    queries_for_select_test_set = dict(queries_dict)
    k_fold = 5
    for i in range(k_fold):
        q_id_train_list = []
        train_X = []
        train_Y = []
        train_TYPES = []

        test_X = []
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

            for train_set in q_id_train_set:
                if train_set[1]=="0":
                    label_zero_count += 1

                    if (label_zero_count<=1):
                        train_X.append(train_set[0])
                        train_Y.append(train_set[1])
                        train_TYPES.append(train_set[2])
                        q_id_train_list.append(query_ids_train)
                else:
                    train_X.append(train_set[0])
                    train_Y.append(train_set[1])
                    train_TYPES.append(train_set[2])
                    q_id_train_list.append(query_ids_train)

        train_Y = pd.get_dummies(train_Y)
        train_Y = train_Y.values.tolist()


        train_X = np.array(train_X)
        train_Y = np.array(train_Y)

        for query_ids_test in queries_for_test_set:
            label_zero_count = 0
            q_id_test_set = trainset_average_w2v[query_ids_test[0]]
            for test_set in q_id_test_set:
                if test_set[1]=="0":
                    label_zero_count += 1

                #if (label_zero_count<=1):
                test_X.append(test_set[0])
                test_Y.append(test_set[1])
                test_TYPES.append(test_set[2])
                q_id_test_list.append(query_ids_test[0])

        test_Y_one_hot = pd.get_dummies(test_Y)
        test_Y_one_hot = test_Y_one_hot.values.tolist()

        test_X = np.array(test_X)
        test_Y_one_hot = np.array(test_Y_one_hot)

        ######################## generate trec output for IR measures :) ########################

        predict_classes_v2, predicted_prob_v2 = model_type_retrieval_v2(train_X, train_Y, test_X, test_Y_one_hot)
        trec_output_modelv2 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v2, predicted_prob_v2)

        predict_classes_v1, predicted_prob_v1 = model_type_retrieval_v1(train_X, train_Y, test_X, test_Y_one_hot)
        trec_output_modelv1 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v1, predicted_prob_v1)

        predict_classes_v3, predicted_prob_v3 = model_type_retrieval_v3(train_X, train_Y, test_X, test_Y_one_hot)
        trec_output_modelv3 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v3, predicted_prob_v3)


        predict_classes_v4, predicted_prob_v4 = model_type_retrieval_v4(train_X, train_Y, test_X, test_Y_one_hot)
        trec_output_modelv4 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v4, predicted_prob_v4)


        ######################## generate trec output for IR measures :) ########################
        print("\n-----------------------------------------------\n\n\n")
        

    trec_output_modelv2 = trec_output_modelv2.rstrip('\n')
    trec_output_modelv1 = trec_output_modelv1.rstrip('\n')
    trec_output_modelv3 = trec_output_modelv3.rstrip('\n')
    trec_output_modelv4 = trec_output_modelv4.rstrip('\n')

    modelv2_path = models_path + "2.run"
    modelv1_path = models_path + "1.run"
    modelv3_path = models_path + "3.run"
    modelv4_path = models_path + "4.run"

    create_file(modelv2_path, trec_output_modelv2)
    create_file(modelv1_path, trec_output_modelv1)
    create_file(modelv3_path, trec_output_modelv3)
    create_file(modelv4_path, trec_output_modelv4)


    # sys.exit(1)