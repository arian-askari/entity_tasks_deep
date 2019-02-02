def get_trec_output_str(trec_ouput_dict):
    delimeter = "	"
    iter_str = "Q0"
    trec_output_str = ""
    for query_id, detailsList in trec_ouput_dict.items():
        detailsList_sorted = sorted(detailsList, key=lambda x: x[2], reverse=True)
        i = 0
        for detail in detailsList_sorted:
            doc_id = detail[0]
            rank_str = str(i)
            sim_score = detail[2]
            run_id = detail[3]
            trec_output_str += query_id + delimeter + iter_str + delimeter + doc_id + delimeter + rank_str + delimeter + sim_score + delimeter + run_id + "\n"
            i += 1
    return trec_output_str


def get_trec_output_classification(q_id_list, test_TYPES, test_Y, predict_classes, predicted_prob):
    trec_output_str = ""
    trec_ouput_dict = dict()

    for q_id_test, q_candidate_type, true_predict, predict_class, predict_prob \
            in zip(q_id_list, test_TYPES, test_Y, predict_classes, predicted_prob):
        # 1
        query_id = q_id_test

        # 2
        iter_str = "Q0"

        # 3
        doc_id = q_candidate_type

        # 4
        rank_str = "0"  # trec az in estefade nemikone, felan ino nemikhad dorost print konam:)

        # 5
        sim_score = (predict_class + 1) * predict_prob[
            predict_class]  # (predict_class+1), baraye inke baraye class 0, score e ehtemal sefr nashe, hamaro +1 kardam, dar kol tasiir nadare, vase class 7 ham 1 mishe va score ha relative mishan
        sim_score = str(sim_score)

        # 6
        run_id = "Model_Deep"

        delimeter = "	"

        if query_id not in trec_ouput_dict:
            trec_ouput_dict[query_id] = [(doc_id, rank_str, sim_score, run_id)]
        else:
            trec_ouput_dict[query_id].append((doc_id, rank_str, sim_score, run_id))

    trec_output_str = get_trec_output_str(trec_ouput_dict)
    return trec_output_str


def get_trec_output_regression(q_id_list, test_TYPES, test_Y, predict_classes):
    trec_output_str = ""
    trec_ouput_dict = dict()

    for q_id_test, q_candidate_type, true_predict, predict_class \
            in zip(q_id_list, test_TYPES, test_Y, predict_classes):
        # 1
        query_id = q_id_test

        # 2
        iter_str = "Q0"

        # 3
        doc_id = q_candidate_type

        # 4
        rank_str = "0"  # trec az in estefade nemikone, felan ino nemikhad dorost print konam:)

        # 5
        sim_score = None
        sim_score = str(predict_class[0])  # model is regression

        # 6
        run_id = "Model_Deep"

        delimeter = "	"

        if query_id not in trec_ouput_dict:
            trec_ouput_dict[query_id] = [(doc_id, rank_str, sim_score, run_id)]
        else:
            trec_ouput_dict[query_id].append((doc_id, rank_str, sim_score, run_id))

    trec_output_str = get_trec_output_str(trec_ouput_dict)
    return trec_output_str
