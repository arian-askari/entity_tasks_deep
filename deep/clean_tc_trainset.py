import os, json
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=False)

trainset_average_w2v = None
trainset_TC = None

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


dirname = os.path.dirname(__file__)

trainset_average_w2v_path = os.path.join(dirname, '../data/types/sig17/trainset_average_w2v_sig17.txt')
trainset_translation_matrix_type_tfidf_terms_path = os.path.join(dirname,'../data/types/sig17/trainset_translation_matrix_tfidf_terms')


def get_trainset_average_w2v():
    print("trainset_average_w2v_path",trainset_average_w2v_path)
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
    return train_set_average_dict


def load_trainset_average_w2v():
    global trainset_average_w2v
    if trainset_average_w2v is None:
        print("trainset_average_w2v loading...")
        trainset_average_w2v = get_trainset_average_w2v()
        print("trainset_average_w2v loaded...")


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

def get_split_data_type_centric(queries_for_train, queries_for_test_set, k):
    global trainset_TC
    global trainset_average_w2v_path

    tmp = trainset_translation_matrix_type_tfidf_terms_path + "_" + str(k) + ".json"
    if trainset_average_w2v_path != tmp:
        global trainset_average_w2v
        trainset_average_w2v = None  # in merge model cause bug :)
        trainset_average_w2v_path = tmp

        if trainset_TC is None:
            load_trainset_average_w2v()
            trainset_TC = trainset_average_w2v
        else:
            trainset_average_w2v = trainset_TC

    return _get_train_testdata(queries_for_train, queries_for_test_set)