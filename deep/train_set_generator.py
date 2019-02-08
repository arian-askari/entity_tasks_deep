import os, subprocess, json, ast, sys, re, random, csv, json, math
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import spatial
import utils.file_utils as file_utils
import utils.list_utils as list_utils

np.set_printoptions(threshold=np.inf)
# np.set_printoptions(precision=3)
np.set_printoptions(suppress=False)
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import utils.elastic as es
import utils.type_retrieval as tp
import utils.entity_retrieval as er_detailed
import utils.preprocess as preprocess

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


dirname = os.path.dirname(__file__)

queries_path = os.path.join(dirname, '../data/dbpedia-v1/queries_type_retrieval.json')
qrel_types_path = os.path.join(dirname, '../data/types/qrels-tti-dbpedia.txt')

##################################################################################################
train_set_row_path = os.path.join(dirname, '../data/types/sig17/train_set_row_sig17.csv')
train_set_feature_path = os.path.join(dirname, '../data/types/sig17/train_set_feature_sig17.csv')

types_unique_raw_path = os.path.join(dirname, '../data/types/sig17/types_unique_raw_sig17.csv')
queries_unique_raw_path = os.path.join(dirname, '../data/types/sig17/quries_unique_row_sig17.txt')

types_unique_feature_path = os.path.join(dirname, '../data/types/sig17/types_unique_feature_sig17.csv')
queries_unique_feature_path = os.path.join(dirname, '../data/types/sig17/quries_unique_feature_sig17.csv')

trainset_average_w2v_path = os.path.join(dirname, '../data/types/sig17/trainset_average_w2v_sig17.txt')

queries_w2v_char_level_path = os.path.join(dirname, '../data/types/sig17/queries_w2v_char_level_feature.csv')
q_ret_100_entities_path = os.path.join(dirname, '../data/types/sig17/q_ret_100_entities.csv')
entity_unique_avg_w2v_path = os.path.join(dirname, '../data/types/sig17/entity_unique_avg_w2v.json')

trainset_translation_matrix_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix.txt')

trainset_translation_matrix_detail_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix_score_e_detail_t_raw_')
trainset_translation_matrix_detail_normal_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix_score_e_detail_t_normal_')

trainset_translation_matrix_escore_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix_score_e_raw_')
trainset_translation_matrix_escore_normal_path = os.path.join(dirname, '../data/types/sig17/trainset_translation_matrix_score_e_normal_')


###
type_terms_raw_path = os.path.join(dirname, '../data/types/sig17/types_unique_terms_sig17.csv')
type_terms_unique_w2v_path = os.path.join(dirname, '../data/types/sig17/type_terms_unique_w2v_path_sig17.csv')
trainset_type_terms_avg_q_avg_w2v_path = os.path.join(dirname,
                                                      '../data/types/sig17/trainset_type_terms_avg_q_avg_w2v_sig17.txt')
###


type_tfidf_sorted_terms_raw_path = os.path.join(dirname, '../data/types/sig17/type_tfidf_sorted_terms_sig17.tsv')
type_dfs_sorted_raw_path = os.path.join(dirname, '../data/types/sig17/type_dfs_sorted_sig17.tsv')

type_w2v_char_level_tfidf_terms_sorted_path = os.path.join(dirname,
                                                           '../data/types/sig17/type_w2v_char_level_tfidf_terms_sorted.json')
type_w2v_char_level_dfs_terms_sorted_path = os.path.join(dirname,
                                                         '../data/types/sig17/type_w2v_char_level_dfs_terms_sorted.json')
trainset_translation_matrix_type_tfidf_terms_path = os.path.join(dirname,
                                                                 '../data/types/sig17/trainset_translation_matrix_tfidf_terms')
trainset_translation_matrix_type_sdf_terms_path = os.path.join(dirname,
                                                               '../data/types/sig17/trainset_translation_matrix_sdf_terms')

###

trainset_cosine_sim_average_w2v_path = os.path.join(dirname,
                                                    '../data/types/sig17/trainset_cosine_sim_average_w2v_sig17.txt')

type_entity_cnt_path = os.path.join(dirname, '../data/types/sig17/types_details_light.tsv')

# trainset_average_w2v_path = trainset_type_terms_avg_q_avg_w2v_path
# trainset_average_w2v_path = trainset_cosine_sim_average_w2v_path


##################################################################################################

# train_set_feature_path = os.path.join(dirname, '../data/types/train_set_feature.csv')
# train_set_row_path = os.path.join(dirname, '../data/types/train_set_row.csv')
#
# types_unique_raw_path = os.path.join(dirname, '../data/types/types_unique_raw.csv')
# queries_unique_raw_path = os.path.join(dirname, '../data/types/quries_unique_row.csv')
#
# types_unique_feature_path = os.path.join(dirname, '../data/types/types_unique_feature.csv')
# queries_unique_feature_path = os.path.join(dirname, '../data/types/quries_unique_feature.csv')
#
# trainset_average_w2v_path = os.path.join(dirname, '../data/types/trainset_average_w2v.txt')
#
# ####
# queries_w2v_char_level_path = os.path.join(dirname, '../data/types/queries_w2v_char_level_feature.csv')
# q_ret_100_entities_path = os.path.join(dirname, '../data/types/q_ret_100_entities.csv')
# entity_unique_avg_w2v_path = os.path.join(dirname, '../data/types/entity_unique_avg_w2v.json')
#
# trainset_translation_matrix_path = os.path.join(dirname, '../data/types/trainset_translation_matrix.txt')

##################################################################################################

try:
    zzzzz = 0
    # os.remove(types_unique_feature_path)
    # os.remove(queries_unique_feature_path)
except:
    pass

word2vec_train_set_path = os.path.join(dirname, '../data/GoogleNews-vectors-negative300.bin')
# word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True, limit=100000) 4

word_vectors = None
trainset_average_w2v = None


# word_vectors = []

def loadWord2Vec():
    global word_vectors
    if word_vectors is None:
        print("w2v loading...")
        word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True)
        print("w2v loaded...")


def get_trainset_average_w2v():
    print("trainset_average_w2v_path",trainset_average_w2v_path)
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
    return train_set_average_dict


def get_trainset_cosine_sim_average_w2v():
    train_set_average_dict = json.load(open(trainset_cosine_sim_average_w2v_path))
    return train_set_average_dict


def load_trainset_average_w2v():
    global trainset_average_w2v
    if trainset_average_w2v is None:
        print("trainset_average_w2v loading...")
        trainset_average_w2v = get_trainset_average_w2v()
        print("trainset_average_w2v loaded...")


def getVector(word):
    loadWord2Vec()
    if word in word_vectors:
        vector = word_vectors[word]
        return vector
    else:
        # print(word + "\t not in w2v google news vocab !")
        return []


def get_vec_several_try(token):
    vec = getVector(token)
    if len(vec) > 0:  # try to find original term, w2c
        return vec

    tmp = token.lower()
    vec = getVector(tmp)
    if len(vec) > 0:  # try to find full lower term, w2c
        return vec

    tmp = token[:1].lower() + token[1:]
    vec = getVector(tmp)
    if len(vec) > 0:  # try to find first character lower term, w2c
        return vec

    tmp = token[:1].upper() + token[1:]
    vec = getVector(tmp)
    if len(vec) > 0:  # try to find first upperCase term, w2c
        return vec

    tmp = token.upper()
    vec = getVector(tmp)
    if len(vec) > 0:  # try to find full upper term, w2c
        return vec

    return []


def get_average_w2v_multi_try(tokens):
    character_level_list = []  # store list of w2v vector(300-D) for each q_word

    for token in tokens:
        token = token.replace("(", "").replace(")", "")
        vec = get_vec_several_try(token)
        if len(vec) > 0:  # try to find original term, w2c
            character_level_list.append(vec)
            continue

        INDEX_TYPE = "dbpedia_2015_10_types"
        tks = es.getTokens(INDEX_TYPE, token)
        char_level_list_temp = []
        for t in tks:
            t = t.split("'")
            t = t[0]
            vec = get_vec_several_try(t)
            if len(vec) > 0:  # try to find original term, w2c
                char_level_list_temp.append(vec)

        if len(char_level_list_temp) > 0:
            char_level_list_temp = [sum(x) for x in zip(*char_level_list_temp)]
            char_level_list_avg = [x / len(char_level_list_temp) for x in char_level_list_temp]
            character_level_list.append(char_level_list_avg)
            continue

    if len(character_level_list) == 0:
        return []

    e_w2v_character_level_list = np.array(character_level_list)
    vector_mn = e_w2v_character_level_list.mean(axis=0)
    return vector_mn


def get_query_character_level_w2v(q_body):
    tokens = q_body.split(" ")

    q_w2v_character_level_list = []  # store list of w2v vector(300-D) for each q_word

    for token in tokens:
        token = token.replace("(", "").replace(")", "")
        vec = get_vec_several_try(token)
        if len(vec) > 0:  # try to find original term, w2c
            q_w2v_character_level_list.append(vec)
            continue

        INDEX_TYPE = "dbpedia_2015_10_types"
        tks = es.getTokens(INDEX_TYPE, token)
        char_level_list_temp = []
        for t in tks:
            t = t.split("'")
            t = t[0]
            vec = get_vec_several_try(t)
            if len(vec) > 0:  # try to find original term, w2c
                char_level_list_temp.append(vec)

        if len(char_level_list_temp) > 0:
            char_level_list_temp = [sum(x) for x in zip(*char_level_list_temp)]
            char_level_list_avg = [x / len(char_level_list_temp) for x in char_level_list_temp]
            # char_level_list_temp.append(char_level_list_avg)
            q_w2v_character_level_list.append(char_level_list_avg)
            continue

        q_w2v_character_level_list.append(np.zeros(300).tolist())

    return q_w2v_character_level_list


def get_type_character_level_w2v(type, k=100):
    tokens = list_utils.unique_preserve_order(type)

    types_have_vec = ""
    t_w2v_character_level_list = []  # store list of w2v vector(300-D) for each q_word
    cnt_type_have_vec = 0

    for token in tokens:
        if cnt_type_have_vec == k:
            return (types_have_vec, t_w2v_character_level_list)

        token = token.replace("(", "").replace(")", "")
        vec = get_vec_several_try(token)
        if len(vec) > 0:  # try to find original term, w2c
            t_w2v_character_level_list.append(vec.tolist())
            types_have_vec += token + " "
            cnt_type_have_vec += 1
            continue
    return (types_have_vec, t_w2v_character_level_list)


def get_average_w2v(tokens):
    token_resume = 0

    vector = []
    np_array = None

    # print("tokens: ", tokens)
    # print("\n\n")
    # print("tokens len: ", len(tokens))

    while ((len(vector) == 0)):
        if (token_resume == len(tokens)):
            break

        first_token_exist_in_w2v = tokens[token_resume]
        # print("token_resume: ", token_resume)

        vector = getVector(first_token_exist_in_w2v)

        if len(vector) > 0:
            np_array = np.array([np.array(vector)])

        token_resume += 1
    if len(vector) == 0:
        print("tamame token haye query dar w2v nabudand ! :(, tokens:", tokens)
        return np.zeros(300)  # kare ghalati vali vase inke ta akhar run ejra beshe felan!

    for token in tokens[token_resume:]:
        # print('token ', token)
        vector = getVector(token)
        if len(vector) > 0:
            tmp_array = np.array(vector)
            np_array = np.concatenate((np_array, [tmp_array]))

    vector_mn = np_array.mean(axis=0)  # to take the mean of each row, 300D vec
    return vector_mn


def get_qrel_types():
    return get_q_types(qrel_types_path)


def get_q_types(path):
    qrels_types_dict = dict()

    with open(path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also use delimiter="\t" rather than giving a dialect.
            query_id = line[0]
            query_target_type = line[1]
            rel_level = line[2]
            # print(line)
            if query_id not in qrels_types_dict:
                qrels_types_dict[query_id] = [(query_target_type, rel_level)]
            elif query_id in qrels_types_dict:
                qrels_types_dict[query_id].append((query_target_type, rel_level))

    return qrels_types_dict


def raw_train_set_generator():
    # train_set_raw_str = "q_id,q_body,t,t_rel\n"
    train_set_raw_str = ""
    qrels_types_dict = get_qrel_types()

    print(qrels_types_dict)
    print("\n\n")
    with open(queries_path, 'r') as ff:
        queries_json = json.load(ff)

        queries_dict = dict(queries_json)

        for q_id, q_value in dict(queries_dict).items():
            q_types_rel = qrels_types_dict[q_id]

            top_types = tp.retrieve_types(q_value)
            for type_retrieved in top_types:
                type_, score, rank = type_retrieved

                l = len([i for i, v in enumerate(q_types_rel) if v[0] == type_])
                if (l < 1):
                    qrels_types_dict[q_id].append((type_, 0))

        for key, value in qrels_types_dict.items():
            q_id = key

            if q_id not in queries_dict:
                print("q_id not in queries_dict: ", q_id)
                continue

            q_body = queries_dict[q_id]
            q_body = q_body.replace("\t", " ")
            for q_id_types in value:
                type_, relClass = q_id_types
                train_set_raw_str += str(q_id) + "	" + str(q_body) + "	" + str(type_) + "	" + str(
                    relClass) + "\n"
                # break

        f_train_set_row = open(train_set_row_path, 'w')
        f_train_set_row.write(train_set_raw_str)
        f_train_set_row.close()


def get_type_avg_w2v(type_name):
    print("average on type: ", type_name)
    INDEX_TYPE = "dbpedia_2015_10_types"

    results = es.find_by_id(INDEX_TYPE, type_name, True)
    tokens = results['term_vectors']['content']['terms'].keys()
    tokens = list(tokens)
    tokens = [token for token in tokens if not isfloat(token)]

    type_avg_w2v = get_average_w2v(tokens)
    return type_avg_w2v


def get_type_terms_avg_w2v(type_terms):  # terms of type, different with entities abstract !
    print("average on terms of type: ", type_terms)
    tokens = type_terms.split(" ")
    t_terms_avg_w2v = get_average_w2v_multi_try(tokens)
    return t_terms_avg_w2v


def get_query_avg_w2v(q_body):
    INDEX_TYPE = "dbpedia_2015_10_types"
    # tokens = es.getTokens(INDEX_TYPE, q_body)
    tokens = q_body.split(" ")
    q_avg_w2v = get_average_w2v(tokens)
    return q_avg_w2v


def get_entity_avg_w2v(e_abstract):
    INDEX_TYPE = "dbpedia_2015_10_types"
    tokens = es.getTokens(INDEX_TYPE, e_abstract)
    # tokens = e_abstract.split(" ")
    e_avg_w2v = get_average_w2v_multi_try(tokens)
    return e_avg_w2v


def w2v_train_set_generator():
    # str_query_types = "q_id,q_body_avg_w2v,q_type_avg_w2v,rel_class\n"
    # str_query_types = ""
    with open(train_set_row_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])
            q_type = str(line[2])
            q_type_rel_class = str(line[3])

            q_body_avg_w2v = get_query_avg_w2v(q_body)
            q_type_avg_w2v = get_type_avg_w2v(
                q_type)  # mishe in ro dar ram negah daram be ezaye har type ! ta mojadadn hesab nashe agar oon type residim! vali hala felan nemidonam ram am mikeshe ya na va storage chetor mishe va bikhialesh misham ! vali khob mikeshe mage chanta bordar 300 bodi mikhad beshe ! bikhialesh misham hala felan :|

            str_query_types = str(q_id) + "\t" + str(q_body_avg_w2v.tolist()) + "\t" + str(
                q_type_avg_w2v.tolist()) + "\t" + str(q_type_rel_class) + "\n"

            print(q_id)
            print(str_query_types)

            f_train_set_feature = open(train_set_feature_path, 'a+')
            f_train_set_feature.write(str_query_types)
            f_train_set_feature.close()


def types_avg_w2v_generator():
    with open(types_unique_raw_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])

            q_type_avg_w2v = get_type_avg_w2v(
                q_type)  # mishe in ro dar ram negah daram be ezaye har type ! ta mojadadn hesab nashe agar oon type residim! vali hala felan nemidonam ram am mikeshe ya na va storage chetor mishe va bikhialesh misham ! vali khob mikeshe mage chanta bordar 300 bodi mikhad beshe ! bikhialesh misham hala felan :|
            str_types_feature = str(q_type) + "\t" + str(q_type_avg_w2v.tolist()) + "\n"

            print(str_types_feature)

            f_train_set_feature = open(types_unique_feature_path, 'a+')
            f_train_set_feature.write(str_types_feature)
            f_train_set_feature.close()


def quries_avg_w2v_generator():
    with open(queries_unique_raw_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])

            q_body_avg_w2v = get_query_avg_w2v(q_body)

            str_query_features = str(q_id) + "\t" + q_body + "\t" + str(q_body_avg_w2v.tolist()) + "\n"

            print(q_id)
            print(str_query_features)

            f_train_set_feature = open(queries_unique_feature_path, 'a+')
            f_train_set_feature.write(str_query_features)
            f_train_set_feature.close()


def q_w2v_char_level_generator():
    with open(queries_unique_raw_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])

            q_body_w2v_char_level = get_query_character_level_w2v(q_body)

            str_query_features = str(q_id) + "\t" + q_body + "\t" + str(np.array(q_body_w2v_char_level).tolist()) + "\n"

            print(q_id)
            print(str_query_features)

            f_train_set_feature = open(queries_w2v_char_level_path, 'a+')
            f_train_set_feature.write(str_query_features)
            f_train_set_feature.close()


# arian
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def get_type_sorted_tfidf_dfs_terms(type_name, k=100):  # <dbo:food>
    INDEX_NAME = "dbpedia_2015_10_types"

    result = es.find_by_id(INDEX_NAME, type_name, True)
    terms = result['term_vectors']['content']['terms']

    terms_score = []  # {"term":(dfs_score, tf.idf_score"}
    # dar nahayat retrieve top-100 har bar bar asas yeki az score ha :)
    cnt_type = 461

    collection_tokens_count = result['term_vectors']['content']["field_statistics"]["sum_ttf"]
    doc_len = sum([term["term_freq"] for term in terms.values()])
    for term_key, term_value in terms.items():
        term_name = str(term_key)

        if hasNumbers(term_name) or len(term_name) < 4:
            continue

        term_idf = math.log(cnt_type / term_value["doc_freq"])
        term_tf = term_value["term_freq"]
        term_cf = term_value["ttf"]

        term_tf_idf_score = term_tf * term_idf
        p_T_t = (term_tf / doc_len)  # ehtemal rokhdad t dar type T
        p_T_not_t = (term_cf - term_tf) / (collection_tokens_count - doc_len)  # ehtemal rokh dad t dar digar type ha
        term_dfs_score = (p_T_t / ((1 - p_T_t) + (p_T_not_t) + 1))

        terms_score.append((term_name, term_dfs_score, term_tf_idf_score))

    terms_sorted_dfs = sorted(terms_score, key=lambda x: x[1], reverse=True)
    terms_sorted_tf_idf = sorted(terms_score, key=lambda x: x[2], reverse=True)
    # terms_sorted_dfs[:10] #top ten+
    return (terms_sorted_tf_idf[:k], terms_sorted_dfs[:k])


def type_w2v_char_level_generator():
    k = 100
    delimeter = "\t"

    type_tfidf_sorted_terms_raw_str = "type\tterms\ttf_idf_sorted_terms\n"  # headers + new line
    type_dfs_sorted_raw_str = "type\tterms\tsdf_sorted_terms\n"  # headers + new line
    type_w2v_char_level_tfidf_terms_sorted_dict = {}  # {type:(type_terms, [tf_idf_sorted_terms w2v char level])}
    type_w2v_char_level_dfs_terms_sorted_dict = {}  # {type:(type_terms, [sdf_sorted_terms w2v char level])}

    with open(type_terms_raw_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])
            q_type_terms = str(line[1])
            q_type_terms = q_type_terms.lower().split(" ")

            terms_sorted_tf_idf, terms_sorted_dfs = get_type_sorted_tfidf_dfs_terms(q_type,
                                                                                    k=1000)  # i[0] term name, i[1] dfs score, i[2] tf idf score

            t_terms_sorted_tf_idf_list = [i[0] for i in terms_sorted_tf_idf]
            t_terms_sorted_dfs_list = [i[0] for i in terms_sorted_dfs]

            for type_term in q_type_terms:  # add type terms first of list tf idf or dfs sorted terms from entity abstracts :)
                t_terms_sorted_tf_idf_list.insert(0, type_term)
                t_terms_sorted_dfs_list.insert(0, type_term)

            type_terms_tf_idf_have_vec, w2v_tf_idf_char_level = get_type_character_level_w2v(t_terms_sorted_tf_idf_list,
                                                                                             k=k)
            type_terms_dfs_have_vec, w2v_dfs_char_level = get_type_character_level_w2v(t_terms_sorted_dfs_list, k=k)

            type_tfidf_sorted_terms_raw_str += q_type + delimeter + (type_terms_tf_idf_have_vec) + "\n"
            type_dfs_sorted_raw_str += q_type + delimeter + (type_terms_dfs_have_vec) + "\n"

            type_w2v_char_level_tfidf_terms_sorted_dict[q_type] = w2v_tf_idf_char_level

            type_w2v_char_level_dfs_terms_sorted_dict[q_type] = w2v_dfs_char_level

        file_utils.write_file(type_tfidf_sorted_terms_raw_path, type_tfidf_sorted_terms_raw_str, force=True)
        file_utils.write_file(type_dfs_sorted_raw_path, type_dfs_sorted_raw_str, force=True)
        file_utils.wirte_json_file(type_w2v_char_level_tfidf_terms_sorted_path,
                                   type_w2v_char_level_tfidf_terms_sorted_dict, force=True)
        file_utils.wirte_json_file(type_w2v_char_level_dfs_terms_sorted_path, type_w2v_char_level_dfs_terms_sorted_dict,
                                   force=True)


def get_type_tfidf_w2v_char_level():
    """dict: {type:[w2v word level]}"""
    with open(type_w2v_char_level_tfidf_terms_sorted_path, 'r') as ff:
        type_tfidf_w2v_char_level_dict = json.load(ff)
        return type_tfidf_w2v_char_level_dict


def get_type_sdf_w2v_char_level():
    """dict: {type:[w2v word level]}"""
    with open(type_w2v_char_level_dfs_terms_sorted_path, 'r') as ff:
        type_dfs_w2v_char_level_dict = json.load(ff)
        return type_dfs_w2v_char_level_dict

def save_translation_matrix_type_terms(score_type="tf_idf", k=100):
    queries_w2v_char_level_dict = get_queries_char_level_w2v_dict()
    # { q_id: (q_body,q_body_w2v_char_level_list_of_list)}
    path_dict = ""
    type_terms_k_top_char_level_w2v = None
    if score_type == "tf_idf":
        type_terms_k_top_char_level_w2v = get_type_tfidf_w2v_char_level()
        path_dict = trainset_translation_matrix_type_tfidf_terms_path + "_" + str(k) + ".json"
    elif score_type == "dfs":
        type_terms_k_top_char_level_w2v = get_type_sdf_w2v_char_level()
        path_dict = trainset_translation_matrix_type_sdf_terms_path + "_" + str(k) + ".json"
    # {type:[w2v word level]}

    # queries_ret_100_entities_dict = get_queries_ret_100_entities_dict()
    # entity_unique_avg_w2v_dict = get_entity_unique_avg_w2v_dict()
    # {entity_name: w2v_abstract_e}

    train_set_translation_matrix_dict = dict()

    with open(train_set_row_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])
            q_type = str(line[2])
            q_type_rel_class = str(line[3])

            translation_matrix_np = get_trainslation_matrix_type_terms(q_id, q_type, queries_w2v_char_level_dict,
                                                                       type_terms_k_top_char_level_w2v, k=k)
            translation_matrix_list = translation_matrix_np.tolist()

            if q_id not in train_set_translation_matrix_dict:
                train_set_translation_matrix_dict[q_id] = [(translation_matrix_list, q_type_rel_class, q_type)]

            else:
                train_set_translation_matrix_dict[q_id].append((translation_matrix_list, q_type_rel_class, q_type))

        # {q_id: [(translation_matrix_list, q_type_rel_class, q_type)]}
        json.dump(train_set_translation_matrix_dict, fp=open(path_dict, 'w'))

#arian 3 arian3
def get_trainset_translation_matrix_score_e_path(type, k): #"detail","detail_normal","e_score","e_score_normal"
    if type=="detail": #e_score * cnt_entities_of_type
        # train_set_translation_matrix_dict = json.load(open(trainset_translation_matrix_detail_path + str(k) + ".json"))
        path = trainset_translation_matrix_detail_path + str(k) + ".json"
        return path

    if type=="detail_normal":
        # train_set_translation_matrix_dict = json.load(open(trainset_translation_matrix_detail_normal_path + str(k) + ".json"))
        path = trainset_translation_matrix_detail_normal_path + str(k) + ".json"
        return path

    if type=="e_score":
        # train_set_translation_matrix_dict = json.load(open(trainset_translation_matrix_escore_path + str(k) + ".json"))
        path = trainset_translation_matrix_escore_path + str(k) + ".json"
        return path

    if type=="e_score_normal":
        # train_set_translation_matrix_dict = json.load(open(trainset_translation_matrix_escore_normal_path + str(k) + ".json"))
        path = trainset_translation_matrix_escore_normal_path + str(k) + ".json"
        return path

def get_trainslation_matrix_score_e(q_id, type, queries_w2v_char_level_dict, queries_ret_100_entities_dict,
                                    entity_unique_avg_w2v_dict, type_ent_cnt_dict, k=100):
    query_max_len = 14
    entity_max_retrieve = k

    w2v_dim_len = 300

    column_size = entity_max_retrieve
    row_size = query_max_len

    translation_matrix_np = np.zeros([row_size, column_size])
    translation_matrix_e_score = np.zeros([row_size, column_size])

    translation_matrix_np_zscore_normal = np.zeros([row_size, column_size])
    translation_matrix_e_score_zscore_normal = np.zeros([row_size, column_size])

    # queries_w2v_char_level_dict,   { q_id: (q_body,q_body_w2v_char_level_list_of_list)}
    # queries_ret_100_entities_dict, {q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}
    # entity_unique_avg_w2v_dict,     {entity_name: w2v_abstract_e}

    current_row = -1

    q_w2v_words = queries_w2v_char_level_dict[q_id][1]
    q_retrieved_entities = queries_ret_100_entities_dict[q_id]

    for q_w2v_word in q_w2v_words:
        current_column = -1

        current_row += 1

        if (all(v == 0 for v in q_w2v_word)):
            continue  # skip this row zeros, because query w2v doesn't exist !

        row_np = np.zeros(column_size)
        row_np_e_score = np.zeros(column_size)

        for retrieved in q_retrieved_entities[:k]:
            current_column += 1

            q_body, retrieved_entity, types_of_retrieved_entity, abstract, relevant_score, rank = retrieved

            if type not in types_of_retrieved_entity:
                continue

            entity_avg_w2v = entity_unique_avg_w2v_dict[retrieved_entity]
            cosine_sim = get_cosine_similarity(q_w2v_word, entity_avg_w2v)

            row_np[current_column] = (relevant_score) * (float(type_ent_cnt_dict[type][0]))
            row_np_e_score[current_column] = relevant_score

        translation_matrix_np[current_row, :] = row_np
        translation_matrix_e_score[current_row, :] = row_np_e_score
    
    translation_matrix_np_zscore_normal = preprocess.get_normalize2D(translation_matrix_np)
    translation_matrix_e_score_zscore_normal = preprocess.get_normalize2D(translation_matrix_e_score)
        
    # cosine_sim_row = np.random.rand(w2v_dim_len)
    # translation_matrix_np[row_number,:] = cosine_sim_row


    return (translation_matrix_np,translation_matrix_np_zscore_normal, translation_matrix_e_score, translation_matrix_e_score_zscore_normal)


def save_translation_matrix_entity_score(k=100):
    queries_w2v_char_level_dict = get_queries_char_level_w2v_dict()
    # { q_id: (q_body,q_body_w2v_char_level_list_of_list)}

    queries_ret_100_entities_dict = get_queries_ret_100_entities_dict()
    # {q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}

    entity_unique_avg_w2v_dict = get_entity_unique_avg_w2v_dict()
    # {entity_name: w2v_abstract_e}

    type_ent_cnt_dict = get_type_entity_cnt_dict()

    train_set_translation_matrix_dict = dict()
    train_set_translation_matrix_normal_zscore_dict = dict()

    train_set_translation_matrix_e_score_dict = dict()
    train_set_translation_matrix_e_score_normal_zscore_dict = dict()

    with open(train_set_row_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])

            q_body = str(line[1])
            q_type = str(line[2])
            q_type_rel_class = str(line[3])

            translation_matrix_np, translation_matrix_np_zscore_normal,\
            translation_matrix_e_score, translation_matrix_e_score_zscore_normal\
                = get_trainslation_matrix_score_e(q_id, q_type,
                                                  queries_w2v_char_level_dict,
                                                  queries_ret_100_entities_dict,
                                                  entity_unique_avg_w2v_dict,
                                                  type_ent_cnt_dict, k=k)
            
            translation_matrix_list = translation_matrix_np.tolist()
            translation_matrix_np_zscore_normal = translation_matrix_np_zscore_normal.tolist()
            translation_matrix_e_score = translation_matrix_e_score.tolist()
            translation_matrix_e_score_zscore_normal = translation_matrix_e_score_zscore_normal.tolist()

            if q_id not in train_set_translation_matrix_dict:
                train_set_translation_matrix_dict[q_id] = [(translation_matrix_list, q_type_rel_class, q_type)]
                train_set_translation_matrix_normal_zscore_dict[q_id] = [(translation_matrix_np_zscore_normal, q_type_rel_class, q_type)]

                train_set_translation_matrix_e_score_dict[q_id] = [(translation_matrix_e_score, q_type_rel_class, q_type)]
                train_set_translation_matrix_e_score_normal_zscore_dict[q_id] = [(translation_matrix_e_score_zscore_normal, q_type_rel_class, q_type)]

            else:
                train_set_translation_matrix_dict[q_id].append((translation_matrix_list, q_type_rel_class, q_type))

                train_set_translation_matrix_normal_zscore_dict[q_id].append((translation_matrix_np_zscore_normal, q_type_rel_class, q_type))
                train_set_translation_matrix_e_score_dict[q_id].append((translation_matrix_e_score, q_type_rel_class, q_type))
                train_set_translation_matrix_e_score_normal_zscore_dict[q_id].append((translation_matrix_e_score_zscore_normal, q_type_rel_class, q_type))


        json.dump(train_set_translation_matrix_normal_zscore_dict, fp=open(trainset_translation_matrix_detail_normal_path + str(k) + ".json", 'w'))
        json.dump(train_set_translation_matrix_e_score_normal_zscore_dict, fp=open(trainset_translation_matrix_escore_normal_path + str(k) + ".json", 'w'))
        json.dump(train_set_translation_matrix_dict, fp=open(trainset_translation_matrix_detail_path + str(k) + ".json", 'w'))
        json.dump(train_set_translation_matrix_e_score_dict, fp=open(trainset_translation_matrix_escore_path + str(k) + ".json", 'w'))




def get_trainslation_matrix_type_terms(q_id, type, queries_w2v_char_level_dict, type_terms_k_top_char_level_w2v, k):
    """{q_id: [(translation_matrix_list, q_type_rel_class, q_type)]}"""
    query_max_len = 14
    type_term_to_k_cnt = k

    column_size = type_term_to_k_cnt
    row_size = query_max_len

    translation_matrix_np = np.zeros([row_size, column_size])

    current_row = -1

    q_w2v_words = queries_w2v_char_level_dict[q_id][1]
    t_term_w2c_words = type_terms_k_top_char_level_w2v[type]

    for q_w2v_word in q_w2v_words:
        current_column = -1

        current_row += 1

        if (all(v == 0 for v in q_w2v_word)):
            continue  # skip this row zeros, because query w2v doesn't exist !

        row_np = np.zeros(column_size)

        for w2c_type_term in t_term_w2c_words[:k]:
            current_column += 1
            cosine_sim = get_cosine_similarity(q_w2v_word, w2c_type_term)
            row_np[current_column] = cosine_sim

        translation_matrix_np[current_row, :] = row_np

    # cosine_sim_row = np.random.rand(w2v_dim_len)
    # translation_matrix_np[row_number,:] = cosine_sim_row

    # {q_id: [(translation_matrix_list, q_type_rel_class, q_type)]}
    return translation_matrix_np


def get_trainset_translation_matrix_type_tfidf_terms(k):
    train_set_translation_matrix_dict = json.load(
        open(trainset_translation_matrix_type_tfidf_terms_path + "_" + str(k) + ".json"))
    return train_set_translation_matrix_dict


def get_types_feature_dict():
    types_feature = dict()
    '''
    { type_name: t_avg_w2v)}
    queries_feature['q_id'][1]//get avg w2v of q_id !
    '''
    with open(types_unique_feature_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])

            q_type_avg_w2v = str(line[1])
            q_type_avg_w2v = ast.literal_eval(q_type_avg_w2v)

            types_feature[q_type] = q_type_avg_w2v

    return types_feature


# queries_unique_feature_path = os.path.join(dirname, '../data/types/quries_unique_feature.csv')
def get_queries_feature_dict():
    queries_feature = dict()
    '''
    { q_id: (q_body,q_avg_w2v)}
    queries_feature['q_id'][1]//get avg w2v of q_id !
    '''
    with open(queries_unique_feature_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])

            q_body_avg_w2v = str(line[2])
            q_body_avg_w2v = ast.literal_eval(q_body_avg_w2v)
            queries_feature[q_id] = (q_body, q_body_avg_w2v)

    return queries_feature


def get_raw_trainset_dict():
    raw_trainset_dict = dict()
    '''
        {q_id: (q_body, q_type, q_type_rel_class)}
        raw_trainset_dict['q_id'][0]//get q_body of q_id !
        raw_trainset_dict['q_id'][2]//get q_type_rel_class of q_id !
    '''
    with open(train_set_row_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])
            q_type = str(line[2])
            q_type_rel_class = str(line[3])

            if q_id not in raw_trainset_dict:
                raw_trainset_dict[q_id] = [(q_body, q_type, q_type_rel_class)]
            else:
                raw_trainset_dict[q_id].append((q_body, q_type, q_type_rel_class))

    return raw_trainset_dict


def q_rel_entities_generator():
    q_rel_entities_dict = {}
    '''
        {q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}
    '''
    with open(queries_unique_raw_path) as tsv:
        cnt = 0
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            cnt += 1
            q_id = str(line[0])
            q_body = str(line[1])
            print("cnt:", cnt, " q_body:", q_body)
            q_rel_entities_dict[q_id] = er_detailed.retrieve_entities(q_body, k=100)

    json.dump(q_rel_entities_dict, fp=open(q_ret_100_entities_path, 'w'))


def entity_unique_avg_w2v():
    entity_avg_w2v_dict = {}
    '''
        {entity_name: w2v_abstract_e}
    '''
    with open(q_ret_100_entities_path, 'r') as ff:
        q_rel_entities_dict = json.load(ff)

        for q_id, q_ret in q_rel_entities_dict.items():
            for ret in q_ret:
                q_body, retrieved_entity, types_of_retrieved_entity, abstract, relevant_score, rank = ret
                if retrieved_entity not in entity_avg_w2v_dict:
                    entity_avg_w2v_dict[retrieved_entity] = get_entity_avg_w2v(abstract).tolist()

    json.dump(entity_avg_w2v_dict, fp=open(entity_unique_avg_w2v_path, 'w'))


########################################
def get_queries_char_level_w2v_dict():
    queries_w2v_char_level_dict = dict()
    '''
    { q_id: (q_body,q_body_w2v_char_level_list_of_list)}
    queries_feature['q_id'][1]//get list of list of  w2v of q_id terms !
    '''
    with open(queries_w2v_char_level_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])

            q_body_w2v_char_level = str(line[2])
            q_body_w2v_char_level = ast.literal_eval(q_body_w2v_char_level)
            queries_w2v_char_level_dict[q_id] = (q_body, q_body_w2v_char_level)

    return queries_w2v_char_level_dict


def get_queries_ret_100_entities_dict():
    with open(q_ret_100_entities_path, 'r') as ff:
        queries_ret_100_entities_dict = json.load(ff)
        return queries_ret_100_entities_dict


def get_entity_unique_avg_w2v_dict():
    with open(entity_unique_avg_w2v_path, 'r') as ff:
        entity_unique_avg_w2v_dict = json.load(ff)
        return entity_unique_avg_w2v_dict


# arian 2
def get_type_entity_cnt_dict():
    '''
    { type: (cnt_entities, 1/cnt_entities)}
    '''
    type_entity_cnt_dict = {}
    with open(type_entity_cnt_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            type_name = str(line[0])
            cnt_entities = str(line[1])
            one_div_cnt_entities = str(line[2])
            type_entity_cnt_dict[type_name] = (cnt_entities, one_div_cnt_entities)
    return type_entity_cnt_dict


def save_translation_matrix():
    queries_w2v_char_level_dict = get_queries_char_level_w2v_dict()
    # { q_id: (q_body,q_body_w2v_char_level_list_of_list)}

    queries_ret_100_entities_dict = get_queries_ret_100_entities_dict()
    # {q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}

    entity_unique_avg_w2v_dict = get_entity_unique_avg_w2v_dict()
    # {entity_name: w2v_abstract_e}

    train_set_translation_matrix_dict = dict()

    with open(train_set_row_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])
            q_type = str(line[2])
            q_type_rel_class = str(line[3])

            translation_matrix_np = get_trainslation_matrix(q_id, q_type, queries_w2v_char_level_dict,
                                                            queries_ret_100_entities_dict, entity_unique_avg_w2v_dict)
            translation_matrix_list = translation_matrix_np.tolist()

            if q_id not in train_set_translation_matrix_dict:
                train_set_translation_matrix_dict[q_id] = [(translation_matrix_list, q_type_rel_class, q_type)]
            else:
                train_set_translation_matrix_dict[q_id].append((translation_matrix_list, q_type_rel_class, q_type))

        # {q_id: [(translation_matrix_list, q_type_rel_class, q_type)]}
        json.dump(train_set_translation_matrix_dict, fp=open(trainset_translation_matrix_path, 'w'))


def get_cosine_similarity(q_w2v_word, entity_avg_w2v):
    cosine_sim = 1 - spatial.distance.cosine(q_w2v_word, entity_avg_w2v)
    return cosine_sim


def get_trainslation_matrix(q_id, type, queries_w2v_char_level_dict, queries_ret_100_entities_dict,
                            entity_unique_avg_w2v_dict):
    """{q_id: [(translation_matrix_list, q_type_rel_class, q_type)]}"""
    query_max_len = 14
    entity_max_retrieve = 100

    w2v_dim_len = 300

    column_size = entity_max_retrieve
    row_size = query_max_len

    translation_matrix_np = np.zeros([row_size, column_size])

    # queries_w2v_char_level_dict,   { q_id: (q_body,q_body_w2v_char_level_list_of_list)}
    # queries_ret_100_entities_dict, {q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}
    # entity_unique_avg_w2v_dict,     {entity_name: w2v_abstract_e}

    current_row = -1

    q_w2v_words = queries_w2v_char_level_dict[q_id][1]
    q_retrieved_entities = queries_ret_100_entities_dict[q_id]

    for q_w2v_word in q_w2v_words:
        current_column = -1

        current_row += 1

        if (all(v == 0 for v in q_w2v_word)):
            continue  # skip this row zeros, because query w2v doesn't exist !

        row_np = np.zeros(column_size)

        for retrieved in q_retrieved_entities:
            current_column += 1

            q_body, retrieved_entity, types_of_retrieved_entity, abstract, relevant_score, rank = retrieved

            if type not in types_of_retrieved_entity:
                continue

            entity_avg_w2v = entity_unique_avg_w2v_dict[retrieved_entity]
            cosine_sim = get_cosine_similarity(q_w2v_word, entity_avg_w2v)

            row_np[current_column] = cosine_sim

        translation_matrix_np[current_row, :] = row_np

    # cosine_sim_row = np.random.rand(w2v_dim_len)
    # translation_matrix_np[row_number,:] = cosine_sim_row

    # {q_id: [(translation_matrix_list, q_type_rel_class, q_type)]}
    return translation_matrix_np


def get_trainset_translation_matrix_average_w2v():
    train_set_translation_matrix_dict = json.load(open(trainset_translation_matrix_path))
    return train_set_translation_matrix_dict

########################################3

def save_trainset_average_w2v():
    types_feature_dict = get_types_feature_dict()
    queries_feazture_dict = get_queries_feature_dict()
    raw_trainset_dict = get_raw_trainset_dict()

    train_set_average_dict = dict()
    '''
        {q_id: [(merged_avg_feature, rel_class, type_name)]}
    '''
    for train_set_key, train_set_value_list in raw_trainset_dict.items():
        for train_set_value in train_set_value_list:
            q_id = train_set_key
            q_body = train_set_value[0]
            q_type = train_set_value[1]
            q_type_rel_class = train_set_value[2]

            q_body_w2v_avg_feature = queries_feazture_dict[q_id][1]
            q_type_w2v_avg_feature = types_feature_dict[q_type]

            merged_features = q_body_w2v_avg_feature + q_type_w2v_avg_feature

            if q_id not in train_set_average_dict:
                train_set_average_dict[q_id] = [(merged_features, q_type_rel_class, q_type)]
            else:
                train_set_average_dict[q_id].append((merged_features, q_type_rel_class, q_type))
    json.dump(train_set_average_dict, fp=open(trainset_average_w2v_path, 'w'))
    # json.dump(train_set_average_dict, fp=open(trainset_average_w2v_path, 'w'), indent=4, sort_keys=True)


def save_trainset_cosine_sim_average_w2v_type_terms():
    types_feature_dict = get_type_terms_feature_dict()
    queries_feazture_dict = get_queries_feature_dict()
    raw_trainset_dict = get_raw_trainset_dict()

    train_set_average_dict = dict()
    '''
        {q_id: [(merged_avg_feature, rel_class, type_name)]}
    '''
    for train_set_key, train_set_value_list in raw_trainset_dict.items():
        for train_set_value in train_set_value_list:
            q_id = train_set_key
            q_body = train_set_value[0]
            q_type = train_set_value[1]
            q_type_rel_class = train_set_value[2]

            q_body_w2v_avg_feature = queries_feazture_dict[q_id][1]
            q_type_w2v_avg_feature = types_feature_dict[q_type]

            # merged_features = q_body_w2v_avg_feature + q_type_w2v_avg_feature

            feature = get_cosine_similarity(q_body_w2v_avg_feature, q_type_w2v_avg_feature)
            if q_id not in train_set_average_dict:
                train_set_average_dict[q_id] = [(feature, q_type_rel_class, q_type)]
            else:
                train_set_average_dict[q_id].append((feature, q_type_rel_class, q_type))
    json.dump(train_set_average_dict, fp=open(trainset_cosine_sim_average_w2v_path, 'w'))
    # json.dump(train_set_average_dict, fp=open(trainset_average_w2v_path, 'w'), indent=4, sort_keys=True)


def get_type_terms_feature_dict():
    type_terms_feature = dict()
    '''
    { type_name: t_avg_w2v)}
    queries_feature['type'][1]//get avg w2v of trems of type !
    '''
    with open(type_terms_unique_w2v_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])

            q_type_avg_w2v = str(line[1])
            q_type_avg_w2v = ast.literal_eval(q_type_avg_w2v)

            type_terms_feature[q_type] = q_type_avg_w2v

    return type_terms_feature


def type_terms_avg_w2v_generator():
    with open(type_terms_raw_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type_raw = str(line[0])
            q_type_terms = str(line[1])
            print("q_type_terms", q_type_terms)

            q_type_terms = q_type_terms.lower()

            q_type_avg_w2v = get_type_terms_avg_w2v(q_type_terms)
            str_types_feature = str(q_type_raw) + "\t" + str(q_type_avg_w2v.tolist()) + "\n"

            print(str_types_feature)

            f_train_set_feature = open(type_terms_unique_w2v_path, 'a')
            f_train_set_feature.write(str_types_feature)
            f_train_set_feature.close()


def save_trainset_type_terms_w2v():
    types_feature_dict = get_type_terms_feature_dict()
    queries_feazture_dict = get_queries_feature_dict()
    raw_trainset_dict = get_raw_trainset_dict()

    train_set_average_dict = dict()
    '''
        {q_id: [(merged_avg_feature, rel_class, type_name)]}
    '''
    for train_set_key, train_set_value_list in raw_trainset_dict.items():
        for train_set_value in train_set_value_list:
            q_id = train_set_key
            q_body = train_set_value[0]
            q_type = train_set_value[1]
            q_type_rel_class = train_set_value[2]

            q_body_w2v_avg_feature = queries_feazture_dict[q_id][1]
            q_type_w2v_avg_feature = types_feature_dict[q_type]

            merged_features = q_body_w2v_avg_feature + q_type_w2v_avg_feature

            if q_id not in train_set_average_dict:
                train_set_average_dict[q_id] = [(merged_features, q_type_rel_class, q_type)]
            else:
                train_set_average_dict[q_id].append((merged_features, q_type_rel_class, q_type))
    json.dump(train_set_average_dict, fp=open(trainset_type_terms_avg_q_avg_w2v_path, 'w'))
    # json.dump(train_set_average_dict, fp=open(trainset_average_w2v_path, 'w'), indent=4, sort_keys=True)


# arian 4 arian4
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

def get_train_test_data_translation_matric_entity_centric(queries_for_train, queries_for_test_set, type, k):
    tmp = get_trainset_translation_matrix_score_e_path(type=type, k=k)
    if trainset_average_w2v_path != tmp:
        trainset_average_w2v = None  # in merge model cause bug :)
        global trainset_average_w2v
        if trainset_average_w2v is None:
            global trainset_average_w2v_path
            trainset_average_w2v_path = tmp
            load_trainset_average_w2v()

    return get_train_test_data(queries_for_train, queries_for_test_set)

def get_train_test_data_translation_matric_type_centric(queries_for_train, queries_for_test_set, k):
    tmp = trainset_translation_matrix_type_tfidf_terms_path + "_" + str(k) + ".json"
    if trainset_average_w2v_path != tmp:
        trainset_average_w2v = None  # in merge model cause bug :)
        global trainset_average_w2v
        if trainset_average_w2v is None:
            global trainset_average_w2v_path
            trainset_average_w2v_path = tmp
            load_trainset_average_w2v()

    return get_train_test_data(queries_for_train, queries_for_test_set)

def get_train_test_data(queries_for_train, queries_for_test_set):
    global trainset_average_w2v
    if trainset_average_w2v is None:
        load_trainset_average_w2v()
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

    return (train_X, train_Y, test_X, test_Y_one_hot, q_id_test_list, test_TYPES, np.array(test_Y))

# w2v_train_set_generator()
# types_avg_w2v_generator()

#
# quries_avg_w2v_generator()
# types_avg_w2v_generator()
# q_w2v_char_level_generator()

# save_trainset_average_w2v()
# save_trainset_cosine_sim_average_w2v_type_terms()
# quries_avg_w2v_generator()

# q_w2v_char_level_generator()
#
# q_rel_entities_generator()
# entity_unique_avg_w2v()
#
# save_translation_matrix()
# save_translation_matrix()
# save_translation_matrix_entity_score()
# save_translation_matrix_entity_score(k=5)
# save_translation_matrix_entity_score(k=20)
# save_translation_matrix_entity_score(k=50)
# save_translation_matrix_entity_score(k=100)

# type_terms_avg_w2v_generator()
# save_trainset_type_terms_w2v()

# type_w2v_char_level_generator()
# save_translation_matrix_type_terms(score_type="tf_idf", k=2)
# save_translation_matrix_type_terms(score_type="tf_idf", k=5)
# save_translation_matrix_type_terms(score_type="tf_idf", k=10)
# save_translation_matrix_type_terms(score_type="tf_idf", k=20)
# save_translation_matrix_type_terms(score_type="tf_idf", k=50)
# save_translation_matrix_type_terms(score_type="tf_idf", k=100)
# save_translation_matrix_type_terms(score_type="tf_idf", k=300)

# print("eiffel")
# wrd1 = getVector("eiffel")
# print(wrd1)
#
# print("\n\nEiffel")
# wrd = getVector("Eiffel")
# print(wrd)
#
#
# print("\n\nEIFFEL")
# wrd3 = getVector("EIFFEL")
# print(wrd3)

# raw_train_set_generator()

# type = "<dbo:Food>"
# get_type_avg_w2v(type)

# q = "List of films from the surrealist category"
# tokens = get_query_avg_w2v(q)
