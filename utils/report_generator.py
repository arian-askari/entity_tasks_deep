import os, json, random, sys, csv
import utils.file_utils as file_utils

validations_report_path = os.path.join("../data", "reports", "validation_reports_EC_TC_Merge.csv")
test_report_path = os.path.join("../data", "reports", "test_reports_EC_TC_Merge.csv")
trec_path = os.path.join("../data", "types", "qrels.test")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
delimeter = "\t"


import time
class Report_Generator():
    def __init__(self):
        pass

    def append_validation(self, run_path_validation, result_path_validation, input_name, layers, activations, category, fold_outer_i, loss_train_avg
                          , loss_validation_avg, dropout, acc_train_avg, acc_validation_avg
                          , batch_size, epoch, optimizer, loss_function, top_k, run_path_test=""):


        self.__create_result_file(run_path_validation, result_path_validation)
        n_5_validation = self.__get_n5(result_path_validation)

        header = "sep=\"\\t\"\ninput_name" + delimeter + "layers" + delimeter + "activations" + delimeter +\
                        "category" + delimeter + "fold_outer_i" + delimeter + "loss_train_avg" + delimeter +\
                        "loss_validation_avg" + delimeter + "dropout" + delimeter +\
                        "n5_validation" + delimeter + "acc_train_avg" + delimeter + \
                        "acc_validation_avg" + delimeter + "batch_size" + delimeter +\
                        "epoch" + delimeter + "optimizer" + delimeter + "loss_function" + delimeter + "top_k" + "\n"


        row = input_name + delimeter +\
              layers + delimeter + activations + delimeter +\
              category + delimeter + fold_outer_i + delimeter + loss_train_avg + delimeter +\
              loss_validation_avg + delimeter + dropout + delimeter + n_5_validation + delimeter +\
              acc_train_avg + delimeter + acc_validation_avg + delimeter + batch_size + delimeter +\
              epoch + delimeter + optimizer + delimeter + loss_function + delimeter + top_k  + "\n"

        time.sleep(2)

        self.__check_exist(validations_report_path, header)
        file_utils.append_file(validations_report_path, row)


    def append_test(self, run_path_test, result_path_validation, result_path_test, input_name
                    , layers, activations, category, fold_outer_i, loss_train_avg
                    , loss_validation_avg, dropout, loss_test
                    , acc_test
                    , acc_train_avg, acc_validation_avg
                    , batch_size, epoch, optimizer, loss_function, top_k):


        self.__create_result_file(run_path_test, result_path_test)

        n_5_validation = ""
        if len(result_path_validation)>0:
            n_5_validation = self.__get_n5(result_path_validation)

        n_5_test = self.__get_n5(result_path_test)


        header = "sep=\"\\t\"\ninput_name" + delimeter +\
                 "layers" + delimeter + "activations" + delimeter + "category" +\
                 delimeter + "fold_outer_i" + delimeter + "loss_train_best_model" +\
                 delimeter + "loss_validation_best_model" + delimeter + "dropout" +\
                 delimeter + "n5_test" + delimeter + "n5_validation_best_model" +\
                 delimeter + "loss_test" + delimeter + "acc_test" + delimeter +\
                 "acc_train_avg" + delimeter + "acc_validation_avg" + delimeter +\
                 "batch_size" + delimeter + "epoch" + delimeter +\
                 "optimizer" + delimeter + "loss_function" + delimeter + "top_k" + "\n"

        row = input_name + delimeter + layers +\
              delimeter + activations + delimeter + category + delimeter +\
              fold_outer_i + delimeter + loss_train_avg + delimeter +\
              loss_validation_avg + delimeter + dropout + delimeter + \
              n_5_test + delimeter + n_5_validation + delimeter +\
              loss_test + delimeter + acc_test + delimeter + acc_train_avg +\
              delimeter + acc_validation_avg + delimeter + batch_size +\
              delimeter + epoch + delimeter + optimizer + delimeter + loss_function + delimeter + top_k + "\n"

        time.sleep(2)

        self.__check_exist(test_report_path, header)
        file_utils.append_file(test_report_path, row)

    def __create_result_file(self, run_path, res_path):
        "Generate Result File from run_path and trec_path "

        # cmd1 = 'cd ' + ROOT_DIR + " && "
        cmd1 = ""
        cmd2 = 'trec_eval -m all_trec "' + trec_path + '" "' + run_path + '" > "' + res_path + '"'
        os.system(cmd1 + cmd2)


    def __check_exist(self, path, header):
        if not os.path.isfile(path):  # if file not exist:
            f = open(path, "w")
            f.write(header)
            f.close()

    def __get_n5(self, result_path):
        with open(result_path) as tsv:
            for line in csv.reader(tsv, dialect="excel-tab"):  # can also
                if len(line) == 3:
                    measure = line[0].strip()
                    if measure == "ndcg_cut_5":
                        n_5 = line[2].strip()
                        return n_5