import os, subprocess, json, ast, sys, re, random, csv


def retrieve_types(query, k=10):  # retrieve top k type for query, with nordlys :)
    query = query.replace("'", "")

    # query  = 'eminem'
    cmd = "python3.6 -m nordlys.services.tti -q '" + query + "'"

    prc = str(subprocess.check_output(cmd, shell=True));
    prc = prc.replace('\n', '').replace('\r', '').replace('\\n', '')
    prc = prc[1:]

    d = json.loads(prc)
    d = ast.literal_eval(d)

    top_types = []
    for result_key, result_detail in d['results'].items():
        type_ = result_detail['type']

        # type_ = re.findall(r'dbo:(.*?)>', type_)[0]
        score = result_detail['score']
        rank = int(result_key)
        top_types.append((type_, score, rank))

    return top_types