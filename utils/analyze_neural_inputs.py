import os, subprocess, json, ast, sys, re, random, csv, json
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import spatial
import utils.file_utils as file_utils
dirname = os.path.dirname(__file__)

queries_path = os.path.join(dirname, "../data", "dbpedia-v1", "queries_type_retrieval.json")
np.set_printoptions(threshold=np.inf)



def analyze_translation_matrix():
    # trainset_translation_matrix_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix.txt')
    # types_rel_all_values_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_rel_all_values_zero.tsv')
    # types_non_rel_any_values_non_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_non_rel_any_values_non_zero.tsv')
    #
    # types_rel_any_values_non_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_rel_any_values_non_zero.tsv')
    # types_non_rel_all_values_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_non_rel_all_values_zero.tsv')

    trainset_translation_matrix_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix_3d_tope(100topterm(50.json')
    types_rel_all_values_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_rel_all_values_zero3D.tsv')
    types_non_rel_any_values_non_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_non_rel_any_values_non_zero3D.tsv')

    types_rel_any_values_non_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_rel_any_values_non_zero3D.tsv')
    types_non_rel_all_values_zero_path = os.path.join(dirname, '../data/analyze/translation_matrix_types_non_rel_all_values_zero3D.tsv')

    train_set_translation_matrix_dict = json.load(open(trainset_translation_matrix_path))


    cnt_type_total = 0
    cnt_type_rell_translation_zero = 0
    cnt_type_rell_translation_non_zero = 0
    
    delimeter = "\t"
    
    types_rel_all_values_zero_str = "q_id\tq_body\ttype\trelClass\n"
    types_non_rel_any_values_non_zero_str = "q_id\tq_body\ttype\trelClass\tcnt_e_this_type\n"

    types_rel_any_values_non_zero_str = "q_id\tq_body\ttype\trelClass\tcnt_e\n"
    types_non_rel_all_values_zero_str = "q_id\tq_body\ttype\trelClass\n"

    with open(queries_path, 'r') as ff:
        trec_output_test = ""

        queries_json = json.load(ff)

        queries_dict = dict(queries_json)

        for q_id, q_body in queries_dict.items():
            for instance in train_set_translation_matrix_dict[q_id]:
                cnt_type_total +=1
                if "0" not in instance[1]: # Examine Generate False Negative Train Set!
                    translation_matrix_array = instance[0]
                    translation_matrix_array = np.array(translation_matrix_array)
                    if not np.any(translation_matrix_array): #if all values of translation matrix are zero !
                        """Creation False Negative ! """
                        cnt_type_rell_translation_zero+=1
                        # print("q_id", q_id, "q_body", q_body, "| type", instance[2],"| relClass", instance[1], "|| all values are zero!")
                        types_rel_all_values_zero_str += q_id + delimeter + q_body + delimeter + instance[2] + delimeter + instance[1]  + "\n"
                    else:
                        cnt_e = np.count_nonzero(translation_matrix_array[0]) #entities_have_this_non_rel_type_for_this_q
                        types_rel_any_values_non_zero_str+= q_id + delimeter + q_body + delimeter + instance[2] + delimeter + instance[1]  + delimeter + str(cnt_e) + "\n" 
                        cnt_type_rell_translation_non_zero+=1

                elif "0" in instance[1]: # Examine Generate False Positive Train Set!
                    translation_matrix_array = instance[0]
                    translation_matrix_array = np.array(translation_matrix_array)
                    if np.any(translation_matrix_array): #if any values of translation matrix not zero !
                        """Creation False Positive ! """
                        cnt_e = np.count_nonzero(translation_matrix_array[0]) #entities_have_this_non_rel_type_for_this_q
                        # print("q_id", q_id, "q_body", q_body, "| type", instance[2],
                        #       "| relClass", instance[1], "| cnt_e:", cnt_e,
                        #       "|| at least one values not zero!")

                        types_non_rel_any_values_non_zero_str += q_id + delimeter + q_body + delimeter + instance[2] + delimeter + instance[1] + delimeter + str(cnt_e) + "\n"
                    else:
                        types_non_rel_all_values_zero_str += q_id + delimeter + q_body + delimeter + instance[2] + delimeter + instance[1]  + "\n"

    file_utils.create_file(types_rel_all_values_zero_path, types_rel_all_values_zero_str)
    file_utils.create_file(types_non_rel_any_values_non_zero_path, types_non_rel_any_values_non_zero_str)

    file_utils.create_file(types_rel_any_values_non_zero_path, types_rel_any_values_non_zero_str)
    file_utils.create_file(types_non_rel_all_values_zero_path, types_non_rel_all_values_zero_str)


    print("\n\ncnt_type_total:", cnt_type_total, "cnt_type_rell_translation_zero:", cnt_type_rell_translation_zero,"cnt_type_rell_translation_non_zero:", cnt_type_rell_translation_non_zero)

analyze_translation_matrix()
