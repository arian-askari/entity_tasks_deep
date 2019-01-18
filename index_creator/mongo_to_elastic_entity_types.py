import sys
sys.path.append("../")
from configparser import ConfigParser
import re

import time
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from elasticsearch import helpers



from config import config
# from utils import load_data
# ignore_exact = load_data.get_query_ignore_list()['pred']
# ignore_regex = load_data.get_query_ignore_list()['regex']


def is_uri( value):
    """Returns true if the value is uri. """
    if value.startswith("<") and value.endswith(">"):
        return True
    return False


def resolve_uri(uri):
    """Resolves the URI using a simple heuristic."""
    if is_uri(uri):
        if uri.startswith("<dbpedia:Category") or uri.startswith("<dbpedia:File"):
            return uri[uri.rfind(":") + 1:-1].replace("_", " ")
        elif uri.startswith("<http://en.wikipedia.org"):
            return uri[uri.rfind("/") + 1:-1].replace("_", " ")
        elif not uri.startswith("<http"):  # for <dbpedia:XXX>
            return uri[uri.find(":") + 1:-1].replace("_", " ")
    return uri.replace("<", "").replace(">", "")



first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
def camel_case_split(name):
    s1 = first_cap_re.sub(r'\1 \2', name)
    return all_cap_re.sub(r'\1 \2', s1).lower()


def bulk_insert(es,index,docs):
    actions = []
    for k in docs:
        action = {
            "_index": index,
            "_type": "doc",
            "_id": k,
            "_source": docs[k]
        }
        actions.append(action)

    if len(actions) > 0:
        helpers.bulk(es, actions, request_timeout = 300)


def print_time(start_time):
    t = time.time() - start_time
    print(" --> "+str(int(t//60))+":"+str(int(t%60)))

def main():
    start_time = time.time()
    bulk_size = 2000

    cnf = config.Config()
    client = MongoClient(cnf.cf['mongodb']['server'], int(cnf.cf['mongodb']['port']))
    db = client[cnf.cf['mongodb']['dbpedia_db']]
    entities = db[cnf.cf['mongodb']['dbpedia_col']]

    es = Elasticsearch([cnf.cf['elastic']['server']], port=int(cnf.cf['elastic']['port']))

    index_name = cnf.cf['elastic']['entity_type_db']

    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    conf = {
        "settings": {
            "number_of_shards": 1,
            "analysis": {
                "analyzer": {
                    "stop_en": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            },
            "number_of_replicas": "0",
        },
        "mappings": {
            "doc": {
                "properties": {
                    "type_key_values": {"type": "text", "term_vector": "yes", "analyzer": "whitespace"},
                    "type_values": {"type": "text", "term_vector": "yes", "analyzer": "whitespace"},
                }
            }
        }
    }
    es.indices.create(index=index_name, body=conf)
    print("Index Created")

    cnt = 0
    doc_cnt = 0
    docs = {}
    for entity in entities.find():
        ent_cnt = 0

        vals_key = []
        vals_value = []

        for key,item in entity.items():
            if key =="<rdf:type>":
                for v in item:
                    if v.startswith("<dbo:"):
                        ent_cnt += 1
                        vals_key.append(v)
                        val_clean = camel_case_split(resolve_uri(v))
                        vals_value.append(val_clean)

        if ent_cnt > 0:
            doc_cnt += 1
            temp = {
                "type_key_values": vals_key,
                "type_values" : vals_value
            }
            # es.index(index_name,'doc',temp)
            docs[entity['_id']] = temp #id e document mishavad entity name dar inja.
            if doc_cnt % bulk_size == 0:
                bulk_insert(es, index_name, docs)
                docs = {}
                doc_cnt = 0
                print(cnt, end='')
                print_time(start_time)

        cnt += 1
    bulk_insert(es, index_name, docs)
    client.close()

main()
