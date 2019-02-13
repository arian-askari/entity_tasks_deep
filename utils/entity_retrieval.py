import os, subprocess, json, ast, sys, re, random, csv
from utils import elastic as es
from config import config
from utils import utf8_helper
import utils.preprocess as preprocess
import math
from termcolor import colored

cnf = config.Config()

def get_entity_types(entity_name):
    index_name = cnf.cf['elastic']['entity_type_db']
    results = es.find(index_name, {
        'query': {
            'match': {
                '_id': entity_name
            }
        }

    }, False)

    if results['hits']['total'] == 0:
        return []
    else:
        type_keys_list = results['hits']['hits'][0]["_source"]["type_keys"]
        return type_keys_list

        # type_values_list = results['hits']['hits'][0]["_source"]["type_values"]
        # return [type_keys_list, type_values_list]

def get_entity_sorted_tfidf_dfs_terms(entity, abstract, k=100):  # <dbo:food>
    INDEX_NAME = cnf.cf['elastic']['entity_db']

    result = es.find_by_id(INDEX_NAME, entity, True)
    terms = result['term_vectors']['abstract']['terms']

    terms_score = []  # {"term":(dfs_score, tf.idf_score"}
    # dar nahayat retrieve top-100 har bar bar asas yeki az score ha :)
    cnt_entity = 4641784

    for term_key, term_value in terms.items():
        term_name = str(term_key)

        if term_name.isdigit():
            continue

        if preprocess.is_exist_whole_word(term_name, abstract, case_sensitive=False) == False:
            continue

        term_idf = math.log(cnt_entity / term_value["doc_freq"])
        term_tf = term_value["term_freq"]

        term_tf_idf_score = term_tf * term_idf
        terms_score.append((term_name, term_tf_idf_score))

    terms_sorted_tf_idf = sorted(terms_score, key=lambda x: x[1], reverse=True)
    return terms_sorted_tf_idf[:k]


def get_entity_abstract(entity_name, concate_names =False):
    index_name = cnf.cf['elastic']['entity_db']
    results = es.find(index_name, {
        'query': {
            'match': {
                '_id': entity_name
            }
        }

    }, False)

    if concate_names == False:
        abstract = results['hits']['hits'][0]["_source"]["abstract"]
        return abstract
    else:
        names = ' '.join(list(results['hits']['hits'][0]["_source"]["names"]))
        abstract = results['hits']['hits'][0]["_source"]["abstract"]
        # names_with_abstract = names + " "+ abstract
        names_with_abstract = abstract

        tokens_ = es.getTokens(index_name, names_with_abstract, analyzer=es.ANALYZER_STANDARD_CASE_SENSITIVE, remove_numbers=True)

        names_with_abstract = " ".join(tokens_)
        return names_with_abstract #alan dg vase tokenize shodan ba split(" ") kamelan okaye :)

def retrieve_entities(query, k=100):  # retrieve top k type for query, with nordlys :)
    query = query.replace("'", "")

    cmd = "python3.6 -m nordlys.services.er -q '" + query + "'"

    prc = str(subprocess.check_output(cmd, shell=True, timeout=100))
    prc = prc[2:-3]
    d = ast.literal_eval(prc)

    top_entities = []
    cnt = 0
    for result_key, result_detail in d['results'].items():
        if len(top_entities) == 100:
            break

        entity = result_detail['entity']
        entity = utf8_helper.get_utf8(entity)
        type_keys_list  = []
        res_types = get_entity_types(entity)
        if len(res_types)>0:
            type_keys_list = res_types
        else:
            print("entity: ",entity," doesn't have any type")

            #shayad be taske query haye bedun type komak kone ! albate faghat shayaad !
            continue #if doesn't have any type, skip this entity ! can't help us for type retrieval ! :)

        abstract = get_entity_abstract(entity, concate_names =True)
        abstract_tf_idf_sorted = get_entity_sorted_tfidf_dfs_terms(entity, abstract, k=100)


        score = result_detail['score']
        rank = int(result_key)
        top_entities.append((query, entity, type_keys_list, abstract, score, rank, abstract_tf_idf_sorted))
    # print(top_entities)
    print("entity count found for this query; ", len(top_entities))

    return top_entities

def retrieve_entities_per_type(query,q_types, k=100):  # retrieve top k type for query, with nordlys :)
    query = query.replace("'", "")

    cmd = "python3.6 -m nordlys.services.er -q '" + query + "'"

    prc = str(subprocess.check_output(cmd, shell=True, timeout=100))
    prc = prc[2:-3]
    d = ast.literal_eval(prc)

    query_types_entities = {}

    cnt = 0
    for result_key, result_detail in d['results'].items():
        #check nemikonim mizarim ta akhar beree :) vali nemizaram bishtar az 100 ta bara hich type i add she :)

        entity = result_detail['entity']
        entity = utf8_helper.get_utf8(entity)
        type_keys_list = []
        res_types = get_entity_types(entity)
        if len(res_types) > 0:
            type_keys_list = res_types
        else:
            # print("entity: ", entity, " doesn't have any type")
            # shayad be taske query haye bedun type komak kone ! albate faghat shayaad !
            continue  # if doesn't have any type, skip this entity ! can't help us for type retrieval ! :)

        abstract = get_entity_abstract(entity, concate_names=True)
        abstract_tf_idf_sorted = get_entity_sorted_tfidf_dfs_terms(entity, abstract, k=100)

        score = result_detail['score']
        rank = int(result_key)


        for type_candidate, type_relevancy in q_types:
            if type_candidate in type_keys_list:
                if type_candidate not in query_types_entities:
                    query_types_entities[type_candidate] = [(query, entity, type_keys_list, abstract, score, rank, abstract_tf_idf_sorted)]
                elif len(query_types_entities[type_candidate]) <= k:
                    query_types_entities[type_candidate].append((query, entity, type_keys_list, abstract, score, rank, abstract_tf_idf_sorted))


    cnt_non_relevant_no_entities = 0
    cnt_relevant_no_entities = 0

    for type_candidate, type_relevancy in q_types:
        if type_candidate not in query_types_entities:
            text = "for type " + str(type_candidate) + "just found 0" + "entities " + " type_relevancy: " + str(type_relevancy)

            if str(type_relevancy) == "0":
                cnt_non_relevant_no_entities +=1
                print(colored(text, "green"))# oza kamelan khube o bar veghf morad
            else:
                cnt_relevant_no_entities += 1
                print(colored(text, "red")) # oh oh che bad !

        elif len(query_types_entities[type_candidate])<100:

            text = "for type " + str(type_candidate) +  "just found " + str(len(query_types_entities[type_candidate])) + "entities " + " type_relevancy: " + str(type_relevancy)
            if str(type_relevancy) == "0":
                print(colored(text, "green"))  # oza kamelan khube o bar veghf morad
            else:
                print(colored(text, "blue")) # oza yekam khatarie

    print("\ncnt_non_relevant_no_entities:", cnt_non_relevant_no_entities , "\n")
    print("\ncnt relevant_no_entities:", cnt_relevant_no_entities , "\n")
    return query_types_entities

# print(retrieve_entities("roman architecure", k=100))
# e_example = "<dbpedia:A_Killing_Affair>"
# get_entity_types(e_example)
# e_example2 = "<dbpedia:12_Years_a_Slave_(score)>"
# get_entity_abstract(e_example2)

# query = "eminem album music"
# query = "eminem"
# r_k = retrieve_entities(query, k=100)
# print(r_k)

# a = get_entity_abstract("<dbpedia:Catherine_Hamlin>", concate_names =True)
# a = get_entity_abstract("<dbpedia:Albert_Einstein>", concate_names =True)
# print(a)
# INDEX_NAME = cnf.cf['elastic']['entity_db']
# tokens = es.getTokens(INDEX_NAME, a, analyzer=es.ANALYZER_STANDARD_CASE_SENSITIVE)
# print("\n\ntokens", tokens, )
# sys.exit(1)