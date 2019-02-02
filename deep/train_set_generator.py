import os, subprocess, json, ast, sys, re, random, csv, json
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import spatial

np.set_printoptions(threshold=np.inf)

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import utils.elastic as es
import utils.type_retrieval as tp
import utils.entity_retrieval as er_detailed

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
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
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

        if len(char_level_list_temp)>0:
            char_level_list_temp = [sum(x) for x in zip(*char_level_list_temp)]
            char_level_list_avg = [x / len(char_level_list_temp) for x in char_level_list_temp]
            # char_level_list_temp.append(char_level_list_avg)
            q_w2v_character_level_list.append(char_level_list_avg)
            continue

        q_w2v_character_level_list.append(np.zeros(300).tolist())

    return q_w2v_character_level_list


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


def types_avgw2v_generator():
    str_query_types = "type_name, avg_w2v\n"
    '''
    faghat oonai k type haye rel o non rel mibashan rooye file benevisam! va gheyre tekrari,
    tu ram negah daram baraye kodom type ha ghablan hesab shode, mojadad hesab nakonam.
    badaan dige faghat un file ro az roo text mikhunam!

    chon hesab kardan w2v va avg rooye tamame token haye yek type,
     ounam chandin bar kheyli tul mikeshe!
    '''


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
            cnt +=1
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

def save_translation_matrix():
    queries_w2v_char_level_dict = get_queries_char_level_w2v_dict()
    #{ q_id: (q_body,q_body_w2v_char_level_list_of_list)}

    queries_ret_100_entities_dict = get_queries_ret_100_entities_dict()
    #{q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}

    entity_unique_avg_w2v_dict = get_entity_unique_avg_w2v_dict()
    # {entity_name: w2v_abstract_e}

    train_set_translation_matrix_dict = dict()

    with open(train_set_row_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_id = str(line[0])
            q_body = str(line[1])
            q_type = str(line[2])
            q_type_rel_class = str(line[3])

            translation_matrix_np = get_trainslation_matrix(q_id, q_type, queries_w2v_char_level_dict , queries_ret_100_entities_dict, entity_unique_avg_w2v_dict)
            translation_matrix_list = translation_matrix_np.tolist()

            if q_id not in train_set_translation_matrix_dict:
                train_set_translation_matrix_dict[q_id] = [(translation_matrix_list, q_type_rel_class, q_type)]
            else:
                train_set_translation_matrix_dict[q_id].append((translation_matrix_list, q_type_rel_class, q_type))

        json.dump(train_set_translation_matrix_dict, fp=open(trainset_translation_matrix_path, 'w'))


def get_cosine_similarity(q_w2v_word, entity_avg_w2v):
    cosine_sim = 1 - spatial.distance.cosine(q_w2v_word, entity_avg_w2v)
    return cosine_sim


def get_trainslation_matrix(q_id, type, queries_w2v_char_level_dict , queries_ret_100_entities_dict, entity_unique_avg_w2v_dict):
    query_max_len = 14
    entity_max_retrieve = 100

    w2v_dim_len = 300

    column_size = entity_max_retrieve
    row_size = query_max_len

    translation_mattix_np = np.zeros([row_size, column_size])

    #queries_w2v_char_level_dict,   { q_id: (q_body,q_body_w2v_char_level_list_of_list)}
    #queries_ret_100_entities_dict, {q_id: [(q_body, retrieved_entity, [types of retrieved entity], abstract, relevant_score, rank)]}
    #entity_unique_avg_w2v_dict,     {entity_name: w2v_abstract_e}

    current_row = -1

    q_w2v_words = queries_w2v_char_level_dict[q_id][1]
    q_retrieved_entities = queries_ret_100_entities_dict[q_id]

    for q_w2v_word in q_w2v_words:
        current_column = -1

        current_row += 1

        if (all(v == 0 for v in q_w2v_word)):
            continue #skip this row zeros, because query w2v doesn't exist !

        row_np = np.zeros(column_size)

        for retrieved in q_retrieved_entities:
            current_column += 1

            q_body, retrieved_entity, types_of_retrieved_entity , abstract, relevant_score, rank = retrieved

            if type not in types_of_retrieved_entity:
                continue

            entity_avg_w2v = entity_unique_avg_w2v_dict[retrieved_entity]
            cosine_sim = get_cosine_similarity(q_w2v_word, entity_avg_w2v)

            row_np[current_column] = cosine_sim

        translation_mattix_np[current_row, :] = row_np

    # cosine_sim_row = np.random.rand(w2v_dim_len)
    # translation_mattix_np[row_number,:] = cosine_sim_row

    return translation_mattix_np

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
# quries_avg_w2v_generator()

# q_w2v_char_level_generator()
#
#q_rel_entities_generator()
# entity_unique_avg_w2v()
#
# save_translation_matrix()
# save_translation_matrix()

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
