import os, subprocess, json, ast, sys, re, random, csv, json
import seaborn as sns
import numpy as np
np.set_printoptions(threshold=np.inf)

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import utils.elastic as es
import utils.type_retrieval as tp

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


dirname = os.path.dirname(__file__)

queries_path = os.path.join(dirname, '../data/dbpedia-v1/queries_type_retrieval.json')
qrel_types_path = os.path.join(dirname, '../data/types/qrels-tti-dbpedia.txt')
train_set_row_path = os.path.join(dirname, '../data/types/train_set_row.csv')
train_set_feature_path = os.path.join(dirname, '../data/types/train_set_feature.csv')

##################################################################################################
types_unique_raw_path = os.path.join(dirname, '../data/types/types_unique_raw.csv')
queries_unique_raw_path = os.path.join(dirname, '../data/types/quries_unique_row.csv')


types_unique_feature_path = os.path.join(dirname, '../data/types/types_unique_feature.csv')
queries_unique_feature_path = os.path.join(dirname, '../data/types/quries_unique_feature.csv')

trainset_average_w2v_path = os.path.join(dirname, '../data/types/trainset_average_w2v.txt')
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

# word_vectors = []

def loadWord2Vec():
    global word_vectors
    if word_vectors is None:
        print("w2v loading...")
        word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True)
        print("w2v loaded...")

def getVector(word):
    loadWord2Vec()
    if word in word_vectors:
        vector = word_vectors[word]
        return vector
    else:
        # print(word + "\t not in w2v google news vocab !")
        return []

def get_average_w2v(tokens):
    token_resume = 0

    vector = []
    np_array = None

    # print("tokens: ", tokens)
    # print("\n\n")
    # print("tokens len: ", len(tokens))

    while( (len(vector) == 0)):
        if(token_resume == len(tokens)):
            break

        first_token_exist_in_w2v = tokens[token_resume]
        # print("token_resume: ", token_resume)

        vector = getVector(first_token_exist_in_w2v)

        if len(vector) > 0:
            np_array = np.array([np.array(vector)])

        token_resume +=1
    if len(vector) == 0:
        print("tamame token haye query dar w2v nabudand ! :(, tokens:", tokens)
        return np.zeros(300) #kare ghalati vali vase inke ta akhar run ejra beshe felan!

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
                train_set_raw_str += str(q_id) + "	" + str(q_body) + "	" + str(type_) + "	" + str(relClass) + "\n"
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
            q_type_avg_w2v = get_type_avg_w2v(q_type) # mishe in ro dar ram negah daram be ezaye har type ! ta mojadadn hesab nashe agar oon type residim! vali hala felan nemidonam ram am mikeshe ya na va storage chetor mishe va bikhialesh misham ! vali khob mikeshe mage chanta bordar 300 bodi mikhad beshe ! bikhialesh misham hala felan :|

            str_query_types = str(q_id) + "\t" + str(q_body_avg_w2v.tolist()) + "\t" + str(q_type_avg_w2v.tolist()) + "\t" + str(q_type_rel_class) + "\n"

            print(q_id)
            print(str_query_types)

            f_train_set_feature = open(train_set_feature_path, 'a+')
            f_train_set_feature.write(str_query_types)
            f_train_set_feature.close()


def types_avg_w2v_generator():
    with open(types_unique_raw_path) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])

            q_type_avg_w2v = get_type_avg_w2v(q_type) # mishe in ro dar ram negah daram be ezaye har type ! ta mojadadn hesab nashe agar oon type residim! vali hala felan nemidonam ram am mikeshe ya na va storage chetor mishe va bikhialesh misham ! vali khob mikeshe mage chanta bordar 300 bodi mikhad beshe ! bikhialesh misham hala felan :|
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
            queries_feature[q_id] = (q_body,q_body_avg_w2v)


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

def save_trainset_average_w2v():
    types_feature_dict = get_types_feature_dict()
    queries_feazture_dict = get_queries_feature_dict()
    raw_trainset_dict =get_raw_trainset_dict()

    train_set_average_dict = dict()
    '''
        {q_id: [(merged_avg_feature, rel_class)]}
    '''
    for train_set_key, train_set_value_list  in raw_trainset_dict.items():
        for train_set_value in train_set_value_list:
            q_id = train_set_key
            q_body = train_set_value[0]
            q_type = train_set_value[1]
            q_type_rel_class = train_set_value[2]

            q_body_w2v_avg_feature = queries_feazture_dict[q_id][1]
            q_type_w2v_avg_feature = types_feature_dict[q_type]

            merged_features = q_body_w2v_avg_feature + q_type_w2v_avg_feature

            if  q_id not in train_set_average_dict:
                train_set_average_dict[q_id] = [(merged_features, q_type_rel_class)]
            else:
                train_set_average_dict[q_id].append((merged_features, q_type_rel_class))
    json.dump(train_set_average_dict, fp=open(trainset_average_w2v_path, 'w'))
    # json.dump(train_set_average_dict, fp=open(trainset_average_w2v_path, 'w'), indent=4, sort_keys=True)

def get_trainset_average_w2v():
    train_set_average_dict = json.load(open(trainset_average_w2v_path))
    return train_set_average_dict


# w2v_train_set_generator()
# types_avg_w2v_generator()
# quries_avg_w2v_generator()
# save_trainset_average_w2v()

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
