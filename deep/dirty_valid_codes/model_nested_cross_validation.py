import os, json, random, sys
import pandas as pd

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor

# base_path = '/content/gdrive/My Drive/Colab Notebooks/'
# base_path = 'C:\\Users\\PerLab\\Desktop\\Arian'
base_path = ''

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

models_path = os.path.join( "data", "runs", "model_v" )
model_path_prefix = "model_"
# models_path_validation = os.path.join(  "data", "runs", "test","adam","")
models_path_validation = os.path.join(  "data", "runs", "validation","adam","")
# models_path_validation = ""
trainset_average_w2v_path = base_path + 'trainset_average_w2v_sig17.txt'


def get_trainset_average_w2v():
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
    return train_set_average_dict

def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))



trainset_average_w2v = get_trainset_average_w2v()

# mv5_l1_neuron_count_candidate = [100, 500, 1000]
# mv5_l2_neuron_count_candidate = [500, 1000, 100]
mv5_l1_neuron_count_candidate = [100]
mv5_l2_neuron_count_candidate = [1000]
mv5_l1_neuron_count = 0
mv5_l2_neuron_count = 0


mv5_l1_dropout_candidate = [0.5]
mv5_l2_dropout_candidate = [0.5]
mv5_l1_dropout = 0
mv5_l2_dropout = 0

epoch_count = 10000

extra_name_path = ""
current_fold = ""

def get_model():
    activation = 'relu'  # or linear
    dropout_rate = 0.1  # or 0.2


    model = Sequential()
    model.add(Dense(1000, input_shape=(600,)))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))


    model.add(Dense(1000))
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.add(Activation('linear'))

    adam = optimizers.Adam(lr=0.0001)

    model.compile(optimizer=adam, loss='mse', metrics=["accuracy"])

    return model

def grid_search_sample():
    # model = get_model()
    model = KerasClassifier(build_fn=get_model(), batch_size=100, epochs=1)

    '''
    cv_results_ : dict of numpy (masked) ndarrays
    '''
    ########################################################
    # Use scikit-learn to grid search
    activation = ['relu']  # tanh softmax, softplus, softsign
    learn_rate = [0.0001] #[0.0001, 0.001]
    dropout_rate = [0.2, 0.3]#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    neurons = [10]
    optimizer =   ['Adam'] #['SGD', 'RMSprop', 'Adam', 'Adamax', 'Nadam']
    ##############################################################
    # grid search epochs, batch size

    epochs = [1] #[5000]  # add 50, 100, 150 etc
    batch_size = [100]  # add 5, 10, 20, 40, 60, 80, 100 etc


    param_grid = dict(epochs=epochs, batch_size=batch_size, activation=activation, learn_rate=learn_rate,neurons=neurons, dropout_rate = dropout_rate, optimizer=optimizer)

    grid = GridSearchCV(cv=KFold(n_splits=3), estimator=model, param_grid=param_grid, n_jobs=8,verbose=10, scoring="neg_mean_squared_error")

    X_fold1 = np.random.rand(6, 600)
    Y_fold1 = np.random.rand(6, 1)

    X_fold2 = np.random.rand(6, 600)
    Y_fold2 = np.random.rand(6, 1)

    X_fold3 = np.random.rand(6, 600)
    Y_fold3 = np.random.rand(6, 1)

    X_fold4 = np.random.rand(6, 600)
    Y_fold4 = np.random.rand(6, 1)

    X_fold5 = np.random.rand(6,600)
    Y_fold5 = np.random.rand(6, 1)

    X_Train = np.random.rand(3,600)
    Y_Train =  np.random.rand(3,1)

    grid_result = grid.fit(X_Train, Y_Train)
    results = grid.cv_results_

    print(results)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



grid_search_sample()


sys.exit(1)
def model_type_retrieval(train_X, train_Y, test_X, test_Y):  # one_layer, count of neuron is count of types!
    print("layer1- neurons: ", mv5_l1_neuron_count, "\t", "layer2- neurons: ", mv5_l2_neuron_count, "\t", "\t epoch count: ", epoch_count)
    global extra_name_v5_path

    extra_name_path = "_L1N(" + str(mv5_l1_neuron_count) + ")"
    extra_name_path += "_L2N(" + str(mv5_l2_neuron_count) + ")"
    extra_name_path += "_epochCount(" + str(epoch_count) + ")"
    # extra_name_path += "_dropL1(" + str(mv5_l1_dropout) + ")"
    # extra_name_path += "_dropL2(" + str(mv5_l2_dropout) + ")"
    extra_name_path += "_fold(" + str(current_fold) + ")"


    model_type_retrieva_v1 = Sequential()
    model_type_retrieva_v1.add(Dense(mv5_l1_neuron_count, input_shape=(600,)))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(mv5_l2_neuron_count))
    model_type_retrieva_v1.add(Activation('relu'))

    model_type_retrieva_v1.add(Dense(1))
    model_type_retrieva_v1.add(Activation('linear'))  # https://datascienceplus.com/keras-regression-based-neural-networks/

    # sgd = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    adam = optimizers.Adam(lr=0.0001)

    # optimizers.
    # model_type_retrieva_v1.compile(optimizer=sgd, loss='mse', metrics=["accuracy"])
    model_type_retrieva_v1.compile(optimizer=adam, loss='mse', metrics=["accuracy"])

    log_file_name = model_path_prefix  + extra_name_path
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

def get_trec_output_str(trec_ouput_dict):
    iter_str = "Q0"
    delimeter = "	"
    trec_output_str = ""
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
    return trec_output_str

def get_trec_output_classification(q_id_list, test_TYPES, test_Y, predict_classes, predicted_prob):
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

    trec_output_str = get_trec_output_str(trec_ouput_dict)

    return trec_output_str


def get_trec_output_logistic_regression(q_id_list, test_TYPES, test_Y, predict_classes):
    trec_ouput_dict = dict()

    for q_id_test, q_candidate_type, true_predict, predict_class in zip(q_id_list, test_TYPES, test_Y, predict_classes):
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

        if query_id not in trec_ouput_dict:
            trec_ouput_dict[query_id] = [(doc_id, rank_str, sim_score, run_id)]
        else:
            trec_ouput_dict[query_id].append((doc_id, rank_str, sim_score, run_id))

    trec_output_str = get_trec_output_str(trec_ouput_dict)
    return trec_output_str

def create_file(path, data):
    f = open(path, 'w')
    f.write(data)
    f.close()



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


def validation_phase():
    with open(queries_path, 'r') as ff:
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

                # predict_classes_v4, predicted_prob_v4 = model_type_retrieval_v4(train_X, train_Y, test_X, test_Y_one_hot)
                # trec_output_modelv4 += get_trec_output(q_id_test_list, test_TYPES, test_Y, predict_classes_v4,
                #                                        predicted_prob_v4)

                train_Y = np.argmax(train_Y, axis=1)
                predict_classes = model_type_retrieval(train_X, train_Y, test_X, test_Y)  # change test_Y_one_hot to test_Y_one_hot, because its regression model
                trec_output_modelv5 += get_trec_output_logistic_regression(q_id_test_list, test_TYPES, test_Y, predict_classes)

                ######################## generate trec output for IR measures :) ########################
                print("\n-----------------------------------------------\n\n\n")
                # break

            trec_output_modelv5 = trec_output_modelv5.rstrip('\n')

            model_file_name = model_path_prefix + extra_name_path
            modelv5_path = models_path_validation + model_file_name + ".run"

            create_file(modelv5_path, trec_output_modelv5)
            # save_metric_result(model_file_name)
            # sys.exit(1)
            break


def train_test_phase():
    with open(queries_path, 'r') as ff:
        trec_output_modelv5 = ""

        queries_json = json.load(ff)

        queries_dict = dict(queries_json)

        queries_for_select_test_set = dict(queries_dict)
        k_fold = 5
        for i in range(k_fold):
            global current_fold
            current_fold = str(i)
            fold_size = int(len(queries_dict) / k_fold)

            queries_for_test_set = None
            if (i + 1) == k_fold:
                queries_for_test_set = list(queries_for_select_test_set.items())
            else:
                queries_for_test_set = random.sample(list(queries_for_select_test_set.items()), fold_size)

            queries_for_select_test_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)

            queries_for_train = substrac_dicts(queries_dict, queries_for_test_set)


            #arain mohem important !!
            '''
            
            inja bayaad inner cross zade beshe! tamam log hash save beshe !
            dar nahayaat behtarin model retireve beshe rooye un hyperemarater haii k ma dadim grid search bezane !
            baad mifamim ba kodom hyper parameter ha rooye validation behtarin amalkard ro dashte
            har bar behtarin model rooye validation ro midim baray test ! dg ham dobare train nemishe un model!
            tamame model haro ham mitonam rooye file save konam vase mohkam kari !
            
                        
            
            Set the verbose parameter in GridSearchCV to a positive number (the greater the number the more detail you will get). For instance:
            
            GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy', verbose=10)  
            
            
            clf = GridSearchCV(estimator=RandomForestClassifier(), 
                    param_grid=parameter_candidates,
                    cv=5,
                    refit=False!,
                    error_score=0,
                    n_jobs=-1)

            '''





            # get Data for train and test set, with query ids
            train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = get_train_test_data(
                queries_for_train, queries_for_test_set)

            ######################## generate trec output for IR measures :) ########################
            train_Y = np.argmax(train_Y, axis=1)
            predict_classes = model_type_retrieval(train_X, train_Y, test_X,  test_Y)  # change test_Y_one_hot to test_Y_one_hot, because its regression model
            trec_output_modelv5 += get_trec_output_logistic_regression(q_id_test_list, test_TYPES, test_Y, predict_classes)


            ######################## generate trec output for IR measures :) ########################
            print("\n-----------------------------------------------\n\n\n")

        trec_output_modelv5 = trec_output_modelv5.rstrip('\n')

        model_file_name = model_path_prefix  + extra_name_path
        modelv5_path = models_path_validation + model_file_name + ".run"

        create_file(modelv5_path, trec_output_modelv5)




#
# for l1_n in mv5_l1_neuron_count_candidate:
#     for l2_n in mv5_l2_neuron_count_candidate:
#         # for l1_d in mv5_l1_dropout_candidate:
#         #     for l2_d in mv5_l2_dropout_candidate:
#         #         mv5_l1_dropout = l1_d
#         #         mv5_l2_dropout = l2_d
#         #         train_test_phase()
#         mv5_l1_neuron_count = l1_n
#         mv5_l2_neuron_count = l2_n
#         validation_phase()

# train_test_phase()