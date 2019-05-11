import os, subprocess, json, ast, sys, re, random, csv, json, math
import numpy as np
import pandas as pd
from scipy import spatial
import utils.file_utils as file_utils
import utils.list_utils as list_utils
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=False)
COUNT_ENTITES_OF_TYPES = 4641784
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

dirname = os.path.dirname(__file__)
trainset_average_w2v_path = ec_input_path = os.path.join('C:\\', 'cygwin64', 'trainset_translation_matrix_2d_qword_')

word_vectors = None
trainset_average_w2v = None
trainset_EC = None
trainset_TC = None

def get_trainset_average_w2v():
    print("trainset_average_w2v_path",trainset_average_w2v_path)
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
    return train_set_average_dict

def load_ec_input():
    global trainset_average_w2v
    if trainset_average_w2v is None:
        print("trainset_average_w2v loading...")
        trainset_average_w2v = get_trainset_average_w2v()
        print("trainset_average_w2v loaded...")

def get_ec_input_path(top_entities=20, top_k_term_per_entity=50):
    path = ec_input_path + "tope(" + str(top_entities) + "topterm(" + str(top_k_term_per_entity) + ".json"
    return path

def _get_train_testdata(queries_for_train, queries_for_test_set):
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
                    t = np.array(train_set[0])
                    # t = t.flatten()

                    train_X.append(t)
                    train_Y.append(train_set[1])
                    train_TYPES.append(train_set[2])
                    q_id_train_list.append(query_ids_train)
            else:
                if all(v == 0 for v in train_set[0]):
                    continue #agar baraye yek type rel hame matrix esh sefr bud ino vorodi nade be neural...noise hast !
                t = np.array(train_set[0])
                # t = t.flatten()
                train_X.append(t)

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
            t = np.array(test_set[0])
            # t = t.flatten()

            test_X.append(t)

            test_Y.append(test_set[1])
            test_TYPES.append(test_set[2])
            q_id_test_list.append(query_ids_test[0])

    test_Y_one_hot = pd.get_dummies(test_Y)
    test_Y_one_hot = test_Y_one_hot.values.tolist()

    test_X = np.array(test_X)
    test_Y_one_hot = np.array(test_Y_one_hot)

    return (train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, np.array(test_Y))

def get_split_data_entity_centric(queries_for_train, queries_for_test_set, top_entities=20, top_k_term_per_entity=50):
    global trainset_EC
    global trainset_average_w2v_path
    tmp = get_ec_input_path(top_entities=top_entities, top_k_term_per_entity=top_k_term_per_entity)
    if trainset_average_w2v_path != tmp:
        global trainset_average_w2v
        trainset_average_w2v = None  # in merge model cause bug :)
        trainset_average_w2v_path = tmp

        if trainset_EC is None:
            load_ec_input()
            trainset_EC = trainset_average_w2v
        else:
            trainset_average_w2v = trainset_EC

    return _get_train_testdata(queries_for_train, queries_for_test_set)

