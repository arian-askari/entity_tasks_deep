import os, json, random, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.layers import *
from utils import trec_output as trec
from utils import file_utils
from deep import train_set_generator as tsg
from deep.model_generator import Model_Generator

base_path = ''

queries_path = os.path.join("../data", "dbpedia-v1", "queries_type_retrieval.json")

models_path = os.path.join("../data", "runs", "")

input_name = "input(abstract_e_avg)_"
input_dim = (600,)
category = "regression"


def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))

# epoch_count = 10000
# layers_for_evaluate = [[1000,1000,1], [1000,1000,100,1]]
# activation = [["relu","relu","linear"],["relu", "relu", "relu", "linear"]]
# dropout_rates = [0.2,0.3,0.4,0.5]

epoch_count = 1
layers_for_evaluate = [[1,1]]
activation_for_evaluate = [["relu", "linear"]]
dropout_rates = [0.1]

def nested_cross_fold_validation():
    with open(queries_path, 'r') as ff:
        trec_output_test = ""

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

            #################################  validation - hyper paramter tuning #################################
            models_during_validation = []

            for layers, activation in zip(layers_for_evaluate, activation_for_evaluate):
                for dropout_rate in dropout_rates:
                    queries_for_select_validation_set = dict(queries_for_train)
                    fold_size = int(len(queries_for_train) / (k_fold - 1))
                    trec_output_validation = ""
                    model_name = ""

                    for j in range(k_fold - 1):
                        current_fold = str(j+1)

                        queries_validation_set = None
                        if (j + 1) == (k_fold - 1):
                            queries_validation_set = list(queries_for_select_validation_set.items())
                        else:
                            queries_validation_set = random.sample(list(queries_for_select_validation_set.items()), fold_size)

                        queries_for_select_validation_set = substrac_dicts(queries_for_select_validation_set, queries_validation_set)

                        queries_train_set = substrac_dicts(queries_for_train, queries_validation_set)
                        train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data(
                            queries_train_set, queries_validation_set)
                        train_Y = np.argmax(train_Y, axis=1)

                        model = Model_Generator(layers=layers, activation=activation, epochs=epoch_count, dropout=dropout_rate,
                                                category=category, learning_rate=0.0001, loss="mse")  # regression sample

                        model_name = input_name + model.get_model_name()

                        log_path = models_path + "T" + str(i+1) + "_V" + str(j+1) + model_name + ".log"

                        model.set_csv_log_path(log_path)

                        ressult_train = model.fit(train_X, train_Y, input_dim)

                        result_validation = model.predict(test_X, test_Y)
                        predict_values = result_validation["predict"]

                        trec_output_validation += trec.get_trec_output_regression(q_id_test_list, test_TYPES, test_Y, predict_values)

                        loss_train, acc_train = ressult_train["train_loss_latest"], ressult_train["train_acc_mean"]
                        loss_validation, acc_validation = result_validation["loss_mean"], result_validation["acc_mean"]

                        models_during_validation.append((model, loss_train,
                                                         acc_train, loss_validation, acc_validation))

                        #inja mishe yek record baraye config rooye in Ti , V, ba in config ha, ezafe konim dar csv e report :) !

                        print("\n-----------------------------------------------\n\n\n")

                    #save validation run file

                    model_path_run_validation = models_path + "T" + str(i+1) + "_V(all)_" + model_name + ".run"

                    trec_output_validation = trec_output_validation.rstrip('\n')
                    file_utils.create_file(model_path_run_validation, trec_output_validation)

            #################### Model Selection, from best model in hyperparameter tuning
            # get Data for test set, with query ids
            _, __, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data(queries_for_train, queries_for_test_set)

            models_sorted = sorted(models_during_validation, key=lambda x: x[3])  # ascending sort

            best_model, loss_train, acc_train, loss_validation, acc_validation = models_sorted[0]
            result_test = best_model.predict(test_X, test_Y)
            loss_test, acc_test = result_test["loss_mean"], result_test["acc_mean"]
            predict_values = result_test["predict"]

            best_model_name = best_model.get_model_name()

            model_test_fold_run_path = "T" + str(i+1) + best_model_name + ".run"
            trec_output_test += trec.get_trec_output_regression(q_id_test_list, test_TYPES, test_Y, predict_values)

            temp = trec_output_test.rstrip('\n')
            file_utils.create_file(model_test_fold_run_path, temp)
            #################### evaluate best model on Test (unseen data :) )
            print("\n-----------------------------------------------\n")

        model_test_all_run_path = input_name + "_" + "T(all)" + "_" + category + ".run"
        trec_output_test = trec_output_test.rstrip('\n')

        file_utils.create_file(model_test_all_run_path, trec_output_test)


nested_cross_fold_validation()