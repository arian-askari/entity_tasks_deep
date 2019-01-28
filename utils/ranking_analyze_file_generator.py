import os, subprocess, json, ast, sys, re, random
import csv

dirname = os.path.dirname(__file__)
train_set_row_path = os.path.join(dirname, '../data/analyze/quries_rel.csv')
run_file_path =  os.path.join(dirname, '../data/analyze/baseline.run')
analyze_file_path =  os.path.join(dirname, '../data/analyze/analyze.csv')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text)]
q_dict_run = {}
with open(run_file_path) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):  # can also
        q_id = str(line[0])
        q_type = str(line[2])
        q_type_rank = str(line[3])
        if q_id not in q_dict_run:
            q_dict_run[q_id] = {}
        q_dict_run[q_id][q_type] = q_type_rank
        '''q_t_rel_dict['INEX_LD-2009022']['<dbo:Food>'] = 21(rank) '''

q_t_rel_dict = {}
q_body_dict = {}
with open(train_set_row_path) as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):  # can also
        q_id = str(line[0])
        q_body = str(line[1])
        q_type = str(line[2])
        q_type_rel_class = str(line[3])

        if q_id not in q_body_dict:
            q_body_dict[q_id] = q_body

        if int(q_type_rel_class)>0:
            if q_id not in q_t_rel_dict:
                q_t_rel_dict[q_id] = {}
            '''q_t_rel_dict['INEX_LD-2009022']['<dbo:Food>'] = 7 '''
            q_t_rel_dict[q_id][q_type] = q_type_rel_class

run_anaylze_out = "q_id,q_body,rel_ranked\n"  # masalan, INEX_LD-2009022, (5) <dbo:Food>,6 \n (1) <dbo:resturant>, 1
for q_id, q_rel_types in q_t_rel_dict.items():
    rel_ranked = "\""
    rel_ranked_list = []
    retrieved_all = ""

    for rel_type, rel_class in q_rel_types.items():
        print(rel_type)
        print(rel_class)
        rank_in_run  = q_dict_run[q_id][rel_type]
        tmp = str(rank_in_run) + "-" + str(rel_type) + " label: " + str(rel_class)
        rel_ranked_list.append(tmp)

    rel_ranked_list.sort(key=natural_keys)

    rel_ranked += "\n".join(rel_ranked_list)
    rel_ranked += "\""
    run_anaylze_out += q_id + ",\"" + q_body_dict[q_id] + "\"," + rel_ranked + "\n"

f = open(analyze_file_path,"w+")
f.write(run_anaylze_out)