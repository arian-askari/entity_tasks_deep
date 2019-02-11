from elasticsearch import Elasticsearch

ANALYZER_STOP = "stop_en"
ANALYZER_WHITE_SPACE = "whitespace"
ANALYZER_STANDARD = "standard"

ANALYZER_STANDARD_CASE_SENSITIVE = "standard_case_sensitive"


es = Elasticsearch(timeout=10000)

def getTokens(index, text, analyzer = ANALYZER_STOP, remove_numbers = False):
    if len(text) == 0:
        return []
    # TODO:: term haii ke faghat adad hastan ro badan bayad hazf konam be nazaram ! ya shayad hamin alan anjam bedam !

    if analyzer == ANALYZER_STANDARD_CASE_SENSITIVE:
        tokens_standard = getTokens(index, text, analyzer= ANALYZER_STANDARD)
        tokens_white_space = getTokens(index, text, analyzer= ANALYZER_WHITE_SPACE)

        i = -1
        for token_stand in tokens_standard:
            i += 1
            for token_white_space in tokens_white_space:
                if token_stand == token_white_space.lower():
                    tokens_standard[i] = token_white_space
        if remove_numbers==False:
            return tokens_standard
        else:
            tokens_standard = [token for token in tokens_standard if token.isdigit()==False]
            return tokens_standard

    else:
        query_tokenied_list = es.indices.analyze(index=index, params={"text": text, "analyzer": analyzer})

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
        result = es.termvectors(index=index, doc_type='doc', id=id, params={"term_statistics":True, "field_statistics":True})
    else:
        result = es.get(index=index, doc_type='doc', id=id)

    return result