import os, json, random, sys
import pandas as pd

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger




# from elasticsearch import Elasticsearch
base_path = '/content/gdrive/My Drive/Colab/'
# base_path = 'C:\\Users\\PerLab\\Desktop\\Arian'
# base_path = ''

ANALYZER_STOP = "stop_en"
# es = Elasticsearch(timeout=10000)


# queries_path = base_path + '\\queries_type_retrieval.json'
queries_path = base_path + 'queries_type_retrieval.json'

# qrel_types_path = base_path + '\\qrels-tti-dbpedia.txt'
qrel_types_path = base_path + 'qrels-tti-dbpedia.txt'
qrels_types_dict = dict()

# word2vec_train_set_path = base_path + '\\GoogleNews-vectors-negative300.bin'
word2vec_train_set_path = base_path + 'GoogleNews-vectors-negative300.bin'
word_vectors = []

# models_path = base_path + '\\data\\runs\\model_v'
# model_path_prefix = "model_v"
# models_path_validation = base_path + '\\data\\runs\\validation\\adam\\'
# trainset_average_w2v_path = base_path + '\\trainset_average_w2v_sig17.txt'

models_path = base_path + os.path.join( "data", "runs", "model_v" )
model_path_prefix = "model_v"
models_path_validation = base_path + os.path.join(  "data", "runs", "validation","adam","")
# models_path_validation = ""
trainset_average_w2v_path = base_path + 'trainset_average_w2v_sig17.txt'

current_fold = ""


def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))


# mv5_l1_neuron_count_candidate = [100, 500, 1000]
mv5_l1_neuron_count_candidate = [100] #for colab
mv5_l2_neuron_count_candidate = [100]
mv5_l3_neuron_count_candidate = [100, 500, 1000]
mv5_l1_neuron_count = 0
mv5_l2_neuron_count = 0
mv5_l3_neuron_count = 0

epoch_count = 10000

extra_name_v5_path = ""


def model_type_retrieval_v5(train_X, train_Y, test_X, test_Y):  # one_layer, count of neuron is count of types!
    print("layer1- neurons: ", mv5_l1_neuron_count, "\t", "layer2- neurons: ", mv5_l2_neuron_count, "\t",
          "layer3- neurons: ", mv5_l3_neuron_count, "\t epoch count: ", epoch_count)
    global extra_name_v5_path

    extra_name_v5_path = "_L1N(" + str(mv5_l1_neuron_count) + ")"
    extra_name_v5_path += "_L2N(" + str(mv5_l2_neuron_count) + ")"
    extra_name_v5_path += "_L3N(" + str(mv5_l3_neuron_count) + ")"
    extra_name_v5_path += "_epochCount(" + str(epoch_count) + ")"
    extra_name_v5_path += "_fold(" + str(current_fold) + ")"

    # https://datascienceplus.com/keras-regression-based-neural-networks/

    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(mv5_l1_neuron_count, input_shape=(600,)))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(mv5_l2_neuron_count))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(mv5_l3_neuron_count))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(1))
    model_type_retrieva_v1.add(Activation('linear'))

    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    adam = optimizers.Adam(lr=0.0001)

    # optimizers.
    # model_type_retrieva_v1.compile(optimizer=sgd, loss='mse', metrics=["accuracy"])
    model_type_retrieva_v1.compile(optimizer=adam, loss='mse', metrics=["accuracy"])

    log_file_name = model_path_prefix + "5" + extra_name_v5_path
    log_path = models_path_validation + log_file_name + ".log"

    # csv_logger = CSVLogger(log_path, append=True, separator=',')
    csv_logger = CSVLogger(log_path, append=False, separator=',')

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=epoch_count, batch_size=100, verbose=2, callbacks=[csv_logger])
    # model_type_retrieva_v1.fit(train_X, train_Y, epochs=epoch_count, batch_size=1, verbose=2)

    # predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    # predicted_prob = model_type_retrieva_v1.predict_proba(test_X)
    # return (predict_classes, predicted_prob)

    predict_classes = model_type_retrieva_v1.predict(test_X)

    return predict_classes


def model_type_retrieval_v4(train_X, train_Y, test_X, test_Y):  # one_layer, count of neuron is count of types!
    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(419, input_shape=(600,)))

    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))

    sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # optimizers.
    model_type_retrieva_v1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=1, batch_size=1, verbose=2)

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    return (predict_classes, predicted_prob)


def model_type_retrieval_v3(train_X, train_Y, test_X, test_Y):
    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(1000, input_shape=(600,)))

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

    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))

    sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    model_type_retrieva_v1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    return (predict_classes, predicted_prob)


def model_type_retrieval_v2(train_X, train_Y, test_X, test_Y):
    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(1000, input_shape=(600,)))

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

    model_type_retrieva_v1.fit(train_X, train_Y, epochs=200, batch_size=1, verbose=2)

    predict_classes = model_type_retrieva_v1.predict_classes(test_X)
    predicted_prob = model_type_retrieva_v1.predict_proba(test_X)

    return (predict_classes, predicted_prob)


def model_type_retrieval_v1(train_X, train_Y, test_X, test_Y):
    avg_q_ = []  # baes bani error e dar nahayat!

    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(20, input_shape=(600,)))

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

    # outupt layer (classifaction model, for NDCG !)
    model_type_retrieva_v1.add(Dense(8))  # classes: 0-7
    model_type_retrieva_v1.add(Activation('softmax'))

    # model_type_retrieva_v1.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    # model_type_retrieva_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    model_type_retrieva_v1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    # train_X = ['q1_type1(600d)','q1_type3(600d)','....','qN_type4(600d)']
    # train_Y = [1, 1, '...', 0]

    # model_type_retrieva_v1.fit(train_X, train_Y, epochs=20, batch_size=1)
    model_type_retrieva_v1.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=0)

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
    # for true_predict, predicted in zip(true_preditc_list, predict_classes):
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
        sim_score = (predict_class + 1) * predict_prob[
            predict_class]  # (predict_class+1), baraye inke baraye class 0, score e ehtemal sefr nashe, hamaro +1 kardam, dar kol tasiir nadare, vase class 7 ham 1 mishe va score ha relative mishan
        sim_score = str(sim_score)

        # 6
        run_id = "Model_Deep"

        delimeter = "	"

        if query_id not in trec_ouput_dict:
            trec_ouput_dict[query_id] = [(doc_id, rank_str, sim_score, run_id)]
        else:
            trec_ouput_dict[query_id].append((doc_id, rank_str, sim_score, run_id))

    for query_id, detailsList in trec_ouput_dict.items():
        detailsList_sorted = sorted(detailsList, key=lambda x: x[2], reverse=True)
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


def get_trec_output_logistic_regression(q_id_list, test_TYPES, test_Y, predict_classes):
    trec_output_str = ""
    trec_ouput_dict = dict()

    for q_id_test, q_candidate_type, true_predict, predict_class \
            in zip(q_id_list, test_TYPES, test_Y, predict_classes):
        # 1
        query_id = q_id_test

        # 2
        iter_str = "Q0"

        # 3
        doc_id = q_candidate_type

        # 4
        rank_str = "0"  # trec az in estefade nemikone, felan ino nemikhad dorost print konam:)

        # 5
        sim_score = None
        sim_score = str(predict_class[0])  # model is regression

        # 6
        run_id = "Model_Deep"

        delimeter = "	"

        if query_id not in trec_ouput_dict:
            trec_ouput_dict[query_id] = [(doc_id, rank_str, sim_score, run_id)]
        else:
            trec_ouput_dict[query_id].append((doc_id, rank_str, sim_score, run_id))

    for query_id, detailsList in trec_ouput_dict.items():
        detailsList_sorted = sorted(detailsList, key=lambda x: x[2], reverse=True)
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


def get_trainset_average_w2v():
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
    return train_set_average_dict


trainset_average_w2v = get_trainset_average_w2v()
i = 0


def get_train_test_data(queries_for_train, queries_for_test_set):
    global trainset_average_w2v
    q_id_train_list = []
    train_X = []
    train_Y = []
    train_TYPES = []

    test_X = []
    test_Y = []
    test_TYPES = []
    q_id_test_list = []

    for query_ids_train in queries_for_train.keys():
        label_zero_count = 0
        q_id_train_set = trainset_average_w2v[query_ids_train]

        for train_set in q_id_train_set:
            if train_set[1] == "0":
                label_zero_count += 1

                if (label_zero_count <= 1):
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
            if test_set[1] == "0":
                label_zero_count += 1

            # if (label_zero_count<=1):
            test_X.append(test_set[0])
            test_Y.append(test_set[1])
            test_TYPES.append(test_set[2])
            q_id_test_list.append(query_ids_test[0])

    test_Y_one_hot = pd.get_dummies(test_Y)
    test_Y_one_hot = test_Y_one_hot.values.tolist()

    test_X = np.array(test_X)
    test_Y_one_hot = np.array(test_Y_one_hot)

    return (train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y)


def save_metric_result(model_file_name):
    result_path = model_file_name.split(".")[0] + ".result"
    run_path = model_file_name + ".run"
    cmd = 'trec_eval -m all_trec "/media/arian/New Volume/Arian/typeRetrieval/data/types/qrels.test" "/media/arian/New Volume/Arian/typeRetrieval/data/runs/validation/adam/' + run_path + '" > "/media/arian/New Volume/Arian/typeRetrieval/data/runs/validation/' + result_path + '"'
    print(cmd)
    os.system(cmd)


def validation_phase():
    with open(queries_path, 'r') as ff:
        # print(avg_q_)
        trec_output_modelv1 = ""
        trec_output_modelv2 = ""
        trec_output_modelv3 = ""
        trec_output_modelv4 = ""
        trec_output_modelv5 = ""

        queries_json = json.load(ff)

        queries_dict = dict(queries_json)

        queries_for_select_test_set = dict(queries_dict)
        k_fold = 5
        for i in range(k_fold):
            fold_size = int(len(queries_dict) / k_fold)

            ###################### remove queries for test from (train and validation set) #################################
            queries_for_test_set = None
            if (i + 1) == k_fold:
                queries_for_test_set = list(queries_for_select_test_set.items())
            else:
                queries_for_test_set = random.sample(list(queries_for_select_test_set.items()), fold_size)

            queries_for_select_test_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)

            queries_for_train = substrac_dicts(queries_dict, queries_for_test_set)
            ######################remove query for test from train and validation set #################################

            #################################  validation and train folds #################################

            queries_for_select_validation_set = dict(queries_for_train)

            for j in range(k_fold - 1):
                global current_fold
                current_fold = str(j)

                fold_size = int(len(queries_for_select_validation_set) / (k_fold - 1))

                queries_validation_set = None
                if (j + 1) == (k_fold - 1):
                    queries_validation_set = list(queries_for_select_validation_set.items())
                else:
                    queries_validation_set = random.sample(list(queries_for_select_validation_set.items()), fold_size)

                # queries_validation_set = random.sample(list(queries_for_select_validation_set.items()), fold_size)
                queries_for_select_validation_set = substrac_dicts(queries_for_select_validation_set,
                                                                   queries_validation_set)

                #                 queries_train_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)
                queries_train_set = substrac_dicts(queries_for_train, queries_validation_set)

                # get Data for train and test set, with query ids
                train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = get_train_test_data(
                    queries_train_set, queries_validation_set)

                ######################## generate trec output for IR measures :) ########################

                # predict_classes_v2, predicted_prob_v2 = model_type_retrieval_v2(train_X, train_Y, test_X, test_Y_one_hot)
                # trec_output_modelv2 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v2,
                #                                        predicted_prob_v2)
                #
                # predict_classes_v1, predicted_prob_v1 = model_type_retrieval_v1(train_X, train_Y, test_X, test_Y_one_hot)
                # trec_output_modelv1 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v1,
                #                                        predicted_prob_v1)
                #
                # predict_classes_v3, predicted_prob_v3 = model_type_retrieval_v3(train_X, train_Y, test_X, test_Y_one_hot)
                # trec_output_modelv3 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v3,
                #                                        predicted_prob_v3)


                # predict_classes_v4, predicted_prob_v4 = model_type_retrieval_v4(train_X, train_Y, test_X, test_Y_one_hot)
                # trec_output_modelv4 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v4,
                #                                        predicted_prob_v4)

                train_Y = np.argmax(train_Y, axis=1)
                predict_classes_v5 = model_type_retrieval_v5(train_X, train_Y, test_X, test_Y)  # change test_Y_one_hot to test_Y_one_hot, because its regression model
                trec_output_modelv5 += get_trec_output_logistic_regression(q_id_test_list, test_TYPES, test_Y, predict_classes_v5)

                ######################## generate trec output for IR measures :) ########################
                print("\n-----------------------------------------------\n\n\n")
                # break

            # trec_output_modelv2 = trec_output_modelv2.rstrip('\n')
            # trec_output_modelv1 = trec_output_modelv1.rstrip('\n')
            # trec_output_modelv3 = trec_output_modelv3.rstrip('\n')
            # trec_output_modelv4 = trec_output_modelv4.rstrip('\n')
            trec_output_modelv5 = trec_output_modelv5.rstrip('\n')

            # modelv2_path = models_path_validation + "2.run"
            # modelv1_path = models_path_validation + "1.run"
            # modelv3_path = models_path_validation + "3.run"
            # modelv4_path = models_path_validation + "4.run"

            model_file_name = model_path_prefix + "5" + extra_name_v5_path
            modelv5_path = models_path_validation + model_file_name + ".run"

            # create_file(modelv2_path, trec_output_modelv2)
            # create_file(modelv1_path, trec_output_modelv1)
            # create_file(modelv3_path, trec_output_modelv3)
            create_file(modelv5_path, trec_output_modelv5)
            # save_metric_result(model_file_name)
            # sys.exit(1)
            break


def train_test_phase():
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
            fold_size = int(len(queries_dict) / k_fold)

            queries_for_test_set = None
            if (i + 1) == k_fold:
                queries_for_test_set = list(queries_for_select_test_set.items())
            else:
                queries_for_test_set = random.sample(list(queries_for_select_test_set.items()), fold_size)

            queries_for_select_test_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)

            queries_for_train = substrac_dicts(queries_dict, queries_for_test_set)

            # get Data for train and test set, with query ids
            train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = get_train_test_data(
                queries_for_train, queries_for_test_set)

            ######################## generate trec output for IR measures :) ########################

            predict_classes_v2, predicted_prob_v2 = model_type_retrieval_v2(train_X, train_Y, test_X, test_Y_one_hot)
            trec_output_modelv2 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v2,
                                                   predicted_prob_v2)

            predict_classes_v1, predicted_prob_v1 = model_type_retrieval_v1(train_X, train_Y, test_X, test_Y_one_hot)
            trec_output_modelv1 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v1,
                                                   predicted_prob_v1)

            predict_classes_v3, predicted_prob_v3 = model_type_retrieval_v3(train_X, train_Y, test_X, test_Y_one_hot)
            trec_output_modelv3 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v3,
                                                   predicted_prob_v3)

            predict_classes_v4, predicted_prob_v4 = model_type_retrieval_v4(train_X, train_Y, test_X, test_Y_one_hot)
            trec_output_modelv4 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v4,
                                                   predicted_prob_v4)

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


# train_test_phase()
for l1_n in mv5_l1_neuron_count_candidate:
    for l2_n in mv5_l2_neuron_count_candidate:
        for l3_n in mv5_l3_neuron_count_candidate:
            mv5_l1_neuron_count = l1_n
            mv5_l2_neuron_count = l2_n
            mv5_l3_neuron_count = l3_n
            validation_phase()