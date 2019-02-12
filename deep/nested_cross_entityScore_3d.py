import os, json, random, sys
# os.environ['CUDA_VISIBLE_DE VICES'] = '-1'

from keras.layers import *

from utils import trec_output as trec
from utils import file_utils
from deep import train_set_generator as tsg
# from deep.model_generator import Model_Generator
from deep.model_generator_EntityScore_3d import Model_Generator
from utils.report_generator import Report_Generator
report = Report_Generator()

queries_path = os.path.join("../data", "dbpedia-v1", "queries_type_retrieval.json")
models_path = os.path.join("../data", "runs", "")
models_path_from_root = os.path.join("./data", "runs", "")

results_path_from_root = os.path.join("./data", "results", "")
results_path= os.path.join("../data", "results", "")
# input_name = "input(abstract_e_avg)_"
k = 100
input_name = "input(cosine_sim_" + str(k) + "dim)_"

# input_dim = (q_token_cnt*k ,)
# input_dim = (1,)
category = "regression"
# category = "classification"


def substrac_dicts(dict1, dict2):
    return dict(set(dict1.items()) - set((dict(dict2)).items()))

# layers_for_evaluate = [[1000, 1000, 1], [1000, 1000, 100, 1]]
# activation_for_evaluate = [["relu","relu","linear"],["relu", "relu", "relu", "linear"]]
# dropout_rates = [0.2, 0.3, 0.4, 0.5]
# loss_function = "mse"
# batch_size = 100
# optimizer = "adam"



''' test '''
epoch_count = 1
# layers_for_evaluate = [[1, 1], [4, 1],[2, 1]]
layers_for_evaluate = [[1, 8], [4, 8],[2, 8]]
# activation_for_evaluate = [["relu",  "linear"], ["relu", "linear"], ["relu", "linear"]]
activation_for_evaluate = [["relu",  "softmax"], ["relu", "softmax"], ["relu", "softmax"]]
batch_size = 100
loss_function = "mse"

def flat_input(list_data):
    list_data = list_data.tolist()
    list_data_flat = []
    for lst in list_data:
        list_data_flat.append(np.array(lst).flatten())

    return np.array(list_data_flat)


def nested_cross_fold_validation():
    loss_train_best_models_total = np.array([])
    acc_validation_best_models_total = np.array([])

    loss_validation_best_models_total = np.array([])
    acc_train_best_models_total = np.array([])  # np.append(number, np_array)

    loss_test_total = np.array([])
    acc_test_total = np.array([])  # np.append(number, np_array)

    with open(queries_path, 'r') as ff:
        trec_output_test = ""

        queries_json = json.load(ff)

        queries_dict = dict(queries_json)

        queries_for_select_test_set = dict(queries_dict)
        k_fold = 5

        # K cross fold validation
        for i in range(k_fold):
            trec_output_test_per_fold = ""

            fold_size = int(len(queries_dict) / k_fold)

            queries_for_test_set = None
            if (i + 1) == k_fold:
                queries_for_test_set = list(queries_for_select_test_set.items())
            else:
                queries_for_test_set = random.sample(list(queries_for_select_test_set.items()), fold_size)

            queries_for_select_test_set = substrac_dicts(queries_for_select_test_set, queries_for_test_set)

            queries_for_train = substrac_dicts(queries_dict, queries_for_test_set)

            ''' hyper paramter tuning '''
            models_during_validation = []

            for layers, activation in zip(layers_for_evaluate, activation_for_evaluate):
                for dropout_rate in dropout_rates:
                    queries_for_select_validation_set = dict(queries_for_train)
                    fold_size = int(len(queries_for_train) / (k_fold - 1))
                    trec_output_validation = ""
                    model_name = ""

                    loss_train_total = np.array([])
                    acc_validation_total = np.array([])

                    loss_validation_total = np.array([])
                    acc_train_total = np.array([]) #  np.append(number, np_array)
                    for j in range(k_fold - 1):
                        queries_validation_set = None
                        if j == (k_fold - 2):
                            queries_validation_set = list(queries_for_select_validation_set.items())
                        else:
                            queries_validation_set = random.sample(list(queries_for_select_validation_set.items()), fold_size)

                        queries_for_select_validation_set = substrac_dicts(queries_for_select_validation_set, queries_validation_set)

                        queries_train_set = substrac_dicts(queries_for_train, queries_validation_set)
                        # train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data_translation_matric_entity_centric(queries_train_set, queries_validation_set, type_matrixEntityScore, k)
                        # train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data_translation_matric_entity_centric_qavg(queries_train_set, queries_validation_set, type_matrixEntityScore, k)
                        train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data_translation_matrix_3d(queries_train_set, queries_validation_set, top_entities=top_entities, top_k_term_per_entity=top_k_term_per_entity, use_tfidf=use_tfidf)

                        if set_input_flat==True:
                            train_X = flat_input(train_X)
                            test_X = flat_input(test_X)

                        if category == "regression":
                            train_Y = np.argmax(train_Y, axis=1)


                        model = Model_Generator(layers=layers, activation=activation, epochs=epoch_count, dropout=dropout_rate,
                                                category=category, learning_rate=learning_rate, loss=loss_function, batch_size=batch_size,
                                                optimizer=optimizer, top_k = k)  # regression sample

                        model_name = model.get_model_name()

                        log_path = models_path + input_name + "T" + str(i+1) + "_V" + str(j+1) + model_name + ".log"

                        model.set_csv_log_path(log_path)

                        if category == "regression":
                            ressult_train = model.fit(train_X, train_Y, input_dim, test_X, test_Y)
                        else:
                            ressult_train = model.fit(train_X, train_Y, input_dim, test_X, test_Y_one_hot)


                        result_validation = None
                        predict_values = None
                        predict_values = None
                        predict_probs = None
                        if category == "regression":
                            result_validation = model.predict(test_X, test_Y)
                            predict_values = result_validation["predict"]
                            trec_output_validation += trec.get_trec_output_regression(q_id_test_list, test_TYPES, test_Y, predict_values)
                        else:
                            result_validation = model.predict(test_X, test_Y_one_hot)
                            predict_values = result_validation["predict"]
                            predict_probs = result_validation["predict_prob"]
                            trec_output_validation += trec.get_trec_output_classification(q_id_test_list, test_TYPES, test_Y, predict_values, predict_probs)

                        loss_train, acc_train = ressult_train["train_loss_latest"], ressult_train["train_acc_mean"]
                        loss_validation, acc_validation = result_validation["loss_mean"], result_validation["acc_mean"]

                        loss_train_total = np.append(loss_train_total, float(loss_train))
                        acc_train_total = np.append(acc_train_total, float(acc_train))

                        loss_validation_total = np.append(loss_validation_total, float(loss_validation))
                        acc_validation_total = np.append(acc_validation_total, float(acc_validation))

                        models_during_validation.append((model, loss_train,
                                                         acc_train, loss_validation, acc_validation, abs(loss_train-loss_validation)))

                        #inja mishe yek record baraye config rooye in Ti , V, ba in config ha, ezafe konim dar csv e report :) !
                        print("\n-----------------------------------------------\n\n\n")

                    loss_train_avg = np.mean(loss_train_total)
                    loss_validation_avg = np.mean(loss_validation_total)

                    acc_train_avg = np.mean(acc_train_total)
                    acc_validation_avg = np.mean(acc_validation_total)

                    model_path_run_validation = models_path + input_name + "T" + str(i+1) + "_V(all)_" + model_name + ".run"

                    trec_output_validation = trec_output_validation.rstrip('\n')
                    file_utils.create_file(model_path_run_validation, trec_output_validation)

                    """ Generate_report validation"""
                    model_path_from_root_run_validation = models_path + input_name + "T" + str(i+1) + "_V(all)_" + model_name + ".run"
                    model_validation_result_from_root_path = results_path + input_name + "T" + str(i+1) + "_V(all)_" + model_name + ".result"
                    model_validation_result_from_root_path = results_path + input_name + "T" + str(i+1) + "_V(all)_" + model_name + ".result"

                    report.append_validation(
                        run_path_validation=model_path_from_root_run_validation
                        , result_path_validation=model_validation_result_from_root_path
                        , input_name=input_name
                        ,layers=str(layers)
                        , activations=str(activation)
                        , category=category
                        , fold_outer_i=str(i+1)
                        , loss_train_avg=str(loss_train_avg)
                        , loss_validation_avg=str(loss_validation_avg)
                        , dropout=str(dropout_rate)
                        , acc_train_avg=str(acc_train_avg)
                        , acc_validation_avg=str(acc_validation_avg)
                        , batch_size=str(batch_size)
                        , epoch=str(epoch_count)
                        , optimizer=optimizer
                        , loss_function=loss_function
                        , top_k= str(k)
                    )

            ''' Evaluate best model on Test (unseen data :) ) '''

            # _, __, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data_translation_matric_entity_centric(queries_for_train, queries_for_test_set, type_matrixEntityScore, k)
            _, __, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, test_Y = tsg.get_train_test_data_translation_matrix_3d(queries_for_train, queries_for_test_set, top_entities=top_entities, top_k_term_per_entity=top_k_term_per_entity, use_tfidf = use_tfidf)

            if set_input_flat == True:
                test_X = flat_input(test_X)
            models_sorted = sorted(models_during_validation, key=lambda x: x[3])  # ascending sort
            # models_sorted = sorted(models_during_validation, key=lambda x: x[5])  # sort by train loss - validation loss (abs value :))



            best_model, loss_train, acc_train, loss_validation, acc_validation, difference_loss_train_loss_validation = models_sorted[0]
            best_model_name = best_model.get_model_name()

            model_test_fold_run_path = models_path + input_name + "_T" + str(i+1) + "(bestModel)_" + best_model_name + ".run"

            #####################################################################################
            result_validation = None
            predict_values = None
            predict_values = None
            predict_probs = None
            result_test = None

            if category == "regression":
                result_test = best_model.predict(test_X, test_Y)
                predict_values = result_test["predict"]
                trec_output_test_per_fold = trec.get_trec_output_regression(q_id_test_list, test_TYPES, test_Y, predict_values)
                trec_output_test += trec_output_test_per_fold

            else:
                result_test = best_model.predict(test_X, test_Y_one_hot)
                predict_values = result_test["predict"]
                predict_probs = result_test["predict_prob"]
                trec_output_test_per_fold = trec.get_trec_output_classification(q_id_test_list, test_TYPES, test_Y, predict_values, predict_probs)
                trec_output_test += trec_output_test_per_fold
                ######################################################################################################

            loss_test, acc_test = result_test["loss_mean"], result_test["acc_mean"]
            temp = trec_output_test_per_fold.rstrip('\n')
            file_utils.create_file(model_test_fold_run_path, temp)

            """ Generate_report test fold"""
            model_test_fold_run_from_root_path = models_path + input_name + "_T" + str(i + 1) + "(bestModel)_" + best_model_name + ".run"
            model_test_fold_result_from_root_path = results_path + input_name + "_T" + str(i + 1) + "(bestModel)_" + best_model_name + ".result"
            model_validation_result_from_root_path = results_path + input_name + "T" + str(i + 1) + "_V(all)_" + best_model_name + ".result"

            report.append_test(
                run_path_test=model_test_fold_run_from_root_path
                , result_path_test=model_test_fold_result_from_root_path
                , result_path_validation=model_validation_result_from_root_path
                , input_name=input_name
                , layers=str(best_model.get_layers())
                , activations=str(best_model.get_activiation())
                , category=best_model.get_category()
                , fold_outer_i=str(i)
                , loss_train_avg=str(loss_train)
                , loss_validation_avg=str(loss_validation)
                , dropout=str(best_model.get_dropout())
                , loss_test=str(loss_test)
                , acc_test=str(loss_test)
                , acc_train_avg=str(acc_train)
                , acc_validation_avg=str(acc_validation)
                , batch_size=str(best_model.get_batch_size())
                , epoch=str(best_model.get_epoch_count())
                , optimizer=str(best_model.get_optimizer())
                , loss_function=str(best_model.get_loss_function())
                , top_k=str(k)
            )

            loss_train_best_models_total = np.append(loss_train_best_models_total, float(loss_train))
            acc_train_best_models_total = np.append(acc_train_best_models_total, float(acc_train))

            loss_validation_best_models_total= np.append(loss_validation_best_models_total, float(loss_validation))
            acc_validation_best_models_total= np.append(acc_validation_best_models_total, float(acc_validation))

            loss_test_total= np.append(loss_test_total, float(loss_test))
            acc_test_total= np.append(acc_test_total, float(acc_test))

            print("\n-----------------------------------------------\n")

        model_test_all_run_path = models_path + input_name + "_T(all)" + "_" + category + ".run"
        trec_output_test = trec_output_test.rstrip('\n')

        file_utils.create_file(model_test_all_run_path, trec_output_test)

        """ Generate_report evaluate test data on best models in validation"""
        model_test_all_run_from_root_path = models_path + input_name + "_T(all)" + "_" + category + ".run"
        model_test_all_result_from_root_path = results_path + input_name + "_T(all)" + "_" + category + ".result"

        report.append_test(
            run_path_test=model_test_all_run_from_root_path
            , result_path_test=model_test_all_result_from_root_path
            , result_path_validation=""
            , input_name=input_name
            , layers=str("")
            , activations=str("")
            , category= ""
            , fold_outer_i= ""
            , loss_train_avg=str(np.mean(loss_train_best_models_total))
            , loss_validation_avg=str(np.mean(loss_validation_best_models_total))
            , dropout=""
            , loss_test=str(np.mean(loss_test_total))
            , acc_test=str(np.mean(acc_test_total))
            , acc_train_avg=str(np.mean(acc_train_best_models_total))
            , acc_validation_avg=str(np.mean(acc_validation_best_models_total))
            , batch_size=""
            , epoch=""
            , optimizer=""
            , loss_function=""
            , top_k=str(k)
        )

#simple test
# layers_for_evaluate_reg = [[100, 1], [500,1] , [1000,1], [1000, 1000, 1], [1000, 1000, 100, 1],  [1000, 1000, 500, 1],  [1000, 1000, 1000, 1]]
# activation_for_evaluate_reg = [["relu","linear"],["relu","linear"],["relu","linear"],["relu", "relu", "linear"],["relu", "relu","relu", "linear"],["relu", "relu","relu", "linear"],["relu", "relu","relu", "linear"]]


# layers_for_evaluate_reg = [[10, 1]]
# activation_for_evaluate_reg = [["relu", "linear"]]

# layers_for_evaluate_reg = [[1000,100,1]]
# activation_for_evaluate_reg = [["relu","relu","linear"]]

layers_for_evaluate_reg = [[2048, 1]]
activation_for_evaluate_reg = [["relu", "linear"]]

dropout_rates = [0]

categories = ["regression"]
layers_for_evaluates = [layers_for_evaluate_reg]
activation_for_evaluates = [activation_for_evaluate_reg]
batch_size = 128

k_values = [10, 5, 100, 2, 300, 5, 20, 50, 100.0]
epoch_count = 1
optimizer = "adam"
learning_rate = 0.0001
q_token_cnt = 1

top_entities= 20 # age ok bood code jadid, 100 esh konam bebinam chi mishe !
top_k_term_per_entity = 10

# type_matrixEntityScore = "detail"
# type_matrixEntityScore = "e_score"

# type_matrixEntityScore = "detail_normal"
# type_matrixEntityScore = "e_score_normal"

type_matrixEntityScore = "cosine_detail"
# type_matrixEntityScore = "cosine_detail_normal"

set_input_flat = False
use_tfidf = True
for cat, act, layers, k_v in zip(categories, activation_for_evaluates, layers_for_evaluates, k_values):
    k = k_v
    input_dim = None
    if set_input_flat:
        input_dim = (q_token_cnt * k,)
    else:
        input_dim = (q_token_cnt, k)
    input_name = "input(" + type_matrixEntityScore + str(k) + "dim)_"

    category = cat
    activation_for_evaluate = act
    layers_for_evaluate = layers
    nested_cross_fold_validation()