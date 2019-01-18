from elasticsearch import Elasticsearch

ANALYZER_STOP = "stop_en"
es = Elasticsearch(timeout=10000)

def getTokens(index, text):
    # TODO:: term haii ke faghat adad hastan ro badan bayad hazf konam be nazaram ! ya shayad hamin alan anjam bedam !
    query_tokenied_list = es.indices.analyze(index=index, params={"text": text, "analyzer": ANALYZER_STOP})
    tokens = []
    for t in sorted(query_tokenied_list["tokens"], key=lambda x: x["position"]):
        tokens.append(t["token"])

    return tokens

def find(index, conditions, term_vector=False):
    if term_vector is True:
        result = es.search(index=index, body=conditions, params={"size": 1000,"term_statistics":True})
    else:
        result = es.search(index=index, body=conditions, params={"size": 1000})

    return result


def find_by_id(index, id, term_vector=False):
    if term_vector is True:
        result = es.termvectors(index=index, doc_type='doc', id=id, params={"term_statistics":True})
    else:
        result = es.search(index=index, doc_type='doc', id=id)

    return result

