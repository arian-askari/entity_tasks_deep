import os, subprocess, json, ast, sys, re, random, csv
from utils import elastic as es
from config import config

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
    type_keys_list = results['hits']['hits'][0]["_source"]["type_key_values"]
    type_values_list = results['hits']['hits'][0]["_source"]["type_values"]
    return (type_keys_list, type_values_list)


    return [] #types of entity e, read from elastic index dbpedia...!

def get_entity_abstract(entity_name):
    index_name = cnf.cf['elastic']['entity_db']
    results = es.find(index_name, {
        'query': {
            'match': {
                '_id': entity_name
            }
        }

    }, False)
    abstract = results['hits']['hits'][0]["_source"]["abstract"]
    return abstract


    return [] #types of entity e, read from elastic index dbpedia...!


def retrieve_entities(query, k=100):  # retrieve top k type for query, with nordlys :)
    query = query.replace("'", "")

    cmd = "python3.6 -m nordlys.services.er -q '" + query + "'"

    prc = str(subprocess.check_output(cmd, shell=True, timeout=100))
    prc = prc[2:-3]
    d = ast.literal_eval(prc)

    top_entities = []
    for result_key, result_detail in d['results'].items():
        entity = result_detail['entity']

        type_keys_list, type_values_list = get_entity_types(entity)
        abstract = get_entity_abstract(entity)

        # type_ = re.findall(r'dbo:(.*?)>', type_)[0]
        score = result_detail['score']
        rank = int(result_key)
        top_entities.append((entity, type_keys_list, abstract, score, rank))

    return top_entities

# e_example = "<dbpedia:A_Killing_Affair>"
# get_entity_types(e_example)
# e_example2 = "<dbpedia:12_Years_a_Slave_(score)>"
# get_entity_abstract(e_example2)

# query = "eminem album music"
# query = "eminem"
# r_k = retrieve_entities(query, k=100)
# print(r_k)