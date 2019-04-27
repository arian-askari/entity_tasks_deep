from keras.models import load_model
from keras.layers import *
from keras.models import *
from keras import optimizers
from keras.callbacks import CSVLogger
from keras.layers import LeakyReLU
from keras import backend as k
from utils.terminate_on_baseline import TerminateOnBaseline


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
import keras.backend as keras_backend
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import json

np.set_printoptions(threshold=np.nan)
weights_path_tc = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman!/TypeCentric/type_centric_models_saved/input(cosine_sim_50dim)_T2_V4_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(50)_regression.log_model.h5"


from sklearn.preprocessing import MinMaxScaler

def get_model_tc():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",
               input_shape=(14,50,1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=16, kernel_size=(10, 10), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=256, kernel_size=(32, 32), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0))

    model.add(Dense(1))
    model.add(Activation("linear"))
    model.add(Dropout(0))

    return model

def get_model_ec():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu",input_shape=(14,400,1)))  # 1.2 know terms of entities importancy
    model.add(MaxPooling2D(pool_size=(1, 5), strides=(1, 5)))  # 2. get max important term five by five

    model.add(Conv2D(filters=64, kernel_size=(14, 4), strides=1, padding="same", activation="relu",))  # 3 know entities importancy

    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4)))  # 4. get entity iportancy by query phrace
    model.add(AveragePooling2D(pool_size=(14, 1), strides=(14, 1)))  # 5 average of entity iportancy on total query

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0))

    model.add(Dense(1))
    model.add(Activation("linear"))
    model.add(Dropout(0))

    return model


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')


    return x

def save_fig_conv_filters(model, layer_index, path):
    print(model.summary())
    conv = model.layers[layer_index]
    conv_weights = conv.get_weights()[0]
    print(conv_weights.shape)

    min_arr = np.amin(conv_weights)
    max_arr = np.amax(conv_weights)

    for i in range(conv_weights.shape[-1]):

        filters = conv_weights[:, :, :, i]#filters of conv1 are filter.shape[0] * filter.shape[1]

        for j in range(filters.shape[-1]):
            filter = conv_weights[:, :, j, i]  # filters of conv1 are filter.shape[0] * filter.shape[1]

            print(filter.shape)

            filter = filter.reshape(filter.shape[0], filter.shape[1])
            filter = filter.astype(float)

            print(filter.shape)

            filter = deprocess_image(filter)
            # print(filter)
            # plt.imshow(filter, vmin=min_arr , vmax=max_arr, interpolation="none", cmap="gray")
            plt.imshow(filter)

            plt.colorbar()
            # plt.axis([0, 9, 0, 9])

            # plt.ylim([9, 0])
            # plt.xlim([9, 0])

            tmp_path = path  + "11filters"  + str(i) + "filter_" + str(j) + ".png"
            plt.savefig(tmp_path, bbox_inches='tight')

            if j==12:
                print(filter)
                sys.exit(1)
            plt.clf()


def save_fig_conv_filters_subplot(model, layer_index, path):
    conv = model.layers[layer_index]
    conv_weights = conv.get_weights()[0]
    print(conv_weights.shape)

    min_arr = np.amin(conv_weights)
    max_arr = np.amax(conv_weights)
    print(min_arr)
    print(max_arr)
    sys.exit(1)
    for i in range(conv_weights.shape[-1]):

        filters = conv_weights[:, :, :, i]#filters of conv1 are filter.shape[0] * filter.shape[1]

        for j in range(filters.shape[-1]):
            filter = conv_weights[:, :, j, i]  # filters of conv1 are filter.shape[0] * filter.shape[1]

            print(filter.shape)

            filter = filter.reshape(filter.shape[0], filter.shape[1])
            filter = filter.astype(float)

            print(filter.shape)

            # print(filter)
            plt.subplot(math.sqrt(conv_weights.shape[-1]), math.sqrt(conv_weights.shape[-1]), i+1)

            plt.imshow(filter, vmin=min_arr , vmax=max_arr, interpolation="none", cmap="gray")
            plt.colorbar()
            # plt.axis([0, 9, 0, 9])

            # plt.ylim([9, 0])
            # plt.xlim([9, 0])

        tmp_path = path  + "all_filters.png"
        plt.savefig(tmp_path, bbox_inches='tight')


def get_type_terms(type, top_k=50):
    with open("./type_tfidf_sorted_terms_sig17.tsv") as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])
            if q_type == type:
                q_type_terms = str(line[1])
                q_type_terms = q_type_terms.lower().split(" ")
                return q_type_terms[:top_k]


def get_type_entities_cnt(type):
    with open("./type_detail_from_mlFeature.tsv") as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"):  # can also
            q_type = str(line[0])
            if q_type == type:
                cnt_entities = int(str(line[1]))
                # q_type_terms = q_type_terms.lower().split(" ")
                return cnt_entities



def show_matrix(xlabels, ylabels, data, title_axis_x, title_axis_y, main_title):
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation="none", cmap="gray")

    # ax.set_xlabel(title_axis_x)
    # ax.set_ylabel(title_axis_y)
    # ax.set_title(main_title)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # ax.set_xticks(ticks=[0,2,4,5,30,50])
    fig.tight_layout()
    # fig.colorbar(im)
    plt.show()

    plt.cla()


def show_matrix_simple(data, title_axis_x, title_axis_y, main_title):
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation="none", cmap="gray")

    ax.set_xlabel(title_axis_x)
    ax.set_ylabel(title_axis_y)
    ax.set_title(main_title)

    # # ax.set_xticks(ticks=[0,2,4,5,30,50])
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    fig.tight_layout()
    fig.colorbar(im)
    plt.show()
    plt.cla()

def get_visualize_tc_conv1(q_id, type, q_body, top_k_type_term = 50):
    #load moadel type centric weights
    weights_path_tc = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman/TypeCentric/type_centric_models_saved/input(cosine_sim_50dim)_T2_V4_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(50)_regression.log_model.h5"
    model = get_model_tc()
    model.load_weights(weights_path_tc)

    #load query terms and type terms
    q_terms = q_body.split(" ")
    type_terms = get_type_terms(type, top_k=top_k_type_term)

    #load type centric matrix for this q_id and type
    train_set_type_centric_top50_dict = json.load(open("./trainset_translation_matrix_tfidf_terms_50.out"))
    q_id_list_types = train_set_type_centric_top50_dict[q_id]
    input_neural = None
    rel_score = None
    for train_set in q_id_list_types:
        if train_set[2] == type:
            input_neural = train_set[0]
            rel_score = train_set[1]
            input_neural = np.array(input_neural)


    #set detail of plot
    title_axis_x = 'type ' + type + ' terms(sorted by tf idf)'
    title_axis_y = 'query terms'
    xlabels = type_terms
    ylabels = q_terms
    main_title = 'Type Centric Matrix Input\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(type) + "\nlabel: " + rel_score)

    #visualize input matrix for q
    print("input matrix for q:", q_id, q_body)
    show_matrix(xlabels, ylabels, input_neural, title_axis_x, title_axis_y, main_title)

    #visualize output matrix of conv1 for q
    print("predit matrix of conv1 output for q:", q_id, q_body)
    input_neural = np.array([input_neural])
    input_neural = np.expand_dims(input_neural, 3) #expand input_neural to four dimension (samples_cnt, rows, columns, channel_cnt)

    model_tc_conv1 = Sequential()
    model_tc_conv1.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",input_shape=(14, 50, 1), weights=model.layers[0].get_weights()))

    predicted_array = model_tc_conv1.predict(input_neural)
    predict = predicted_array[0, :, :, 0]

    main_title =  'Type Centric Matrix Output of Conv2d_1\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(type) + "\nlabel: " + rel_score)


    show_matrix(xlabels, ylabels, predict, title_axis_x, title_axis_y, main_title)

    


def get_visualize_tc_conv2(q_id, type, q_body, top_k_type_term = 50):
    #load moadel type centric weights
    weights_path_tc = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman/TypeCentric/type_centric_models_saved/input(cosine_sim_50dim)_T2_V4_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(50)_regression.log_model.h5"
    model = get_model_tc()
    model.load_weights(weights_path_tc)
    print(model.summary())

    #load query terms and type terms
    q_terms = q_body.split(" ")
    type_terms = get_type_terms(type, top_k=top_k_type_term)

    #load type centric matrix for this q_id and type
    train_set_type_centric_top50_dict = json.load(open("./trainset_translation_matrix_tfidf_terms_50.out"))
    q_id_list_types = train_set_type_centric_top50_dict[q_id]
    input_neural = None
    rel_score = None
    for train_set in q_id_list_types:
        if train_set[2] == type:
            input_neural = train_set[0]
            rel_score = train_set[1]
            input_neural = np.array(input_neural)


    #set detail of plot
    title_axis_x = 'type ' + type + ' terms(sorted by tf idf)'
    title_axis_y = 'query terms'
    xlabels = type_terms
    ylabels = q_terms
    main_title = 'Type Centric Matrix Input\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(type) + "\nlabel: " + rel_score)

    #visualize input matrix for q
    print("input matrix for q:", q_id, q_body)
    # show_matrix_simple(xlabels, ylabels, input_neural, title_axis_x, title_axis_y, main_title)

    #visualize output matrix of conv1 for q
    print("predit matrix of conv1 output for q:", q_id, q_body)
    input_neural = np.array([input_neural])
    input_neural = np.expand_dims(input_neural, 3) #expand input_neural to four dimension (samples_cnt, rows, columns, channel_cnt)

    model_tc_conv1 = Sequential()
    model_tc_conv1.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",input_shape=(14, 50, 1), weights=model.layers[0].get_weights()))
    model_tc_conv1.add(MaxPooling2D(weights=model.layers[1].get_weights()))
    model_tc_conv1.add(Conv2D(weights=model.layers[2].get_weights(), filters=16, kernel_size=(10, 10), strides=(1, 1), padding="same", activation="relu"))

    predicted_array = model_tc_conv1.predict(input_neural)
    predict = predicted_array[0, :, :, 0]

    main_title =  'Type Centric Matrix Output of Conv2d_2\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(type) + "\nlabel: " + rel_score)

    show_matrix_simple(predict, title_axis_x, title_axis_y, main_title)

def get_visualize_tc_conv3(q_id, type, q_body, top_k_type_term = 50):
    #load moadel type centric weights
    weights_path_tc = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman/TypeCentric/type_centric_models_saved/input(cosine_sim_50dim)_T2_V4_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(50)_regression.log_model.h5"
    model = get_model_tc()
    model.load_weights(weights_path_tc)
    print(model.summary())

    #load query terms and type terms
    q_terms = q_body.split(" ")
    type_terms = get_type_terms(type, top_k=top_k_type_term)

    #load type centric matrix for this q_id and type
    train_set_type_centric_top50_dict = json.load(open("./trainset_translation_matrix_tfidf_terms_50.out"))
    q_id_list_types = train_set_type_centric_top50_dict[q_id]
    input_neural = None
    rel_score = None
    for train_set in q_id_list_types:
        if train_set[2] == type:
            input_neural = train_set[0]
            rel_score = train_set[1]
            input_neural = np.array(input_neural)


    #set detail of plot
    title_axis_x = 'type ' + type + ' terms(sorted by tf idf)'
    title_axis_y = 'query terms'
    xlabels = type_terms
    ylabels = q_terms
    main_title = 'Type Centric Matrix Input\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(type) + "\nlabel: " + rel_score)

    #visualize input matrix for q
    print("input matrix for q:", q_id, q_body)
    # show_matrix_simple(xlabels, ylabels, input_neural, title_axis_x, title_axis_y, main_title)

    #visualize output matrix of conv1 for q
    print("predit matrix of conv1 output for q:", q_id, q_body)
    input_neural = np.array([input_neural])
    input_neural = np.expand_dims(input_neural, 3) #expand input_neural to four dimension (samples_cnt, rows, columns, channel_cnt)

    model_tc_conv1 = Sequential()
    model_tc_conv1.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu",input_shape=(14, 50, 1), weights=model.layers[0].get_weights()))
    model_tc_conv1.add(MaxPooling2D(weights=model.layers[1].get_weights()))
    model_tc_conv1.add(Conv2D(weights=model.layers[2].get_weights(), filters=16, kernel_size=(10, 10), strides=(1, 1), padding="same", activation="relu"))
    model_tc_conv1.add(MaxPooling2D())
    model_tc_conv1.add(Conv2D(filters=256, kernel_size=(32, 32), strides=(1, 1), padding="same", activation="relu"))

    predicted_array = model_tc_conv1.predict(input_neural)
    predict = predicted_array[0, :, :, 0]

    main_title =  'Type Centric Matrix Output of Conv2d_2\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(type) + "\nlabel: " + rel_score)

    show_matrix_simple(predict, title_axis_x, title_axis_y, main_title)

q_id_top_100_entities_dict = json.load(open("./q_ret_100_entities_lm.json"))
def get_visualize_ec_conv3(q_id, type, q_body, top_k_type_term = 50):
    global q_id_top_100_entities_dict
    #load moadel type centric weights
    # weights_path_ec = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman!/EntityCentric/input(cosine_detail20dim)_T1_V1_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(20)_regression.log_model.h5"
    weights_path_ec = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman/EntityCentric/models_saved/input(cosine_detail20dim)_T5_V4_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(20)_regression.log_model.h5"
    model = get_model_ec()
    model.load_weights(weights_path_ec)
    print(model.summary())
    #load query terms and type terms
    q_terms = q_body.split(" ")
    type_entities_cnt = get_type_entities_cnt(type)

    #load columns, entities retrieved for this query
    m_type = type

    m_q_id_top_100_entities_dict = q_id_top_100_entities_dict[q_id]
    q_id_entities = [item[1] for item in m_q_id_top_100_entities_dict[:20]]

    q_id_entities_terms_2d = [item[6] for item in m_q_id_top_100_entities_dict[:20]]
    q_id_entities_terms = []
    for entity_terms in q_id_entities_terms_2d:
        for term, tf_idfscore in entity_terms[:20]:
            q_id_entities_terms.append(term)

        if len(entity_terms)<20:
            cnt_loop = 20 - len(entity_terms)
            for i in range(cnt_loop):
                q_id_entities_terms.append(" ")

    # q_id_entities_terms = [item[6][:20] for item in q_id_top_100_entities_dict]
    # q_id_entities_terms = np.array(q_id_entities_terms)
    # print(q_id_entities_terms)
    # print(q_id_entities)

    #load type centric matrix for this q_id and type
    train_set_entity_centric_top20_dict = json.load(open("./trainset_translation_matrix_2d_qword_tope(20topterm(20.out"))
    q_id_list_2d_matrix = train_set_entity_centric_top20_dict[q_id]
    input_neural = None
    rel_score = None
    for train_set in q_id_list_2d_matrix:
        if train_set[2] == type:
            input_neural = train_set[0]
            rel_score = train_set[1]
            # input_neural = np.array(input_neural) * (1/type_entities_cnt)
            input_neural = np.array(input_neural)
            # input_neural = np.array(input_neural)

    # print(q_id_top_100_entities_dict)


    #set detail of plot
    title_axis_x = 'entities terms(sorted by lm)'
    title_axis_y = 'query terms'
    # xlabels = q_id_entities_terms

    # q_id_entities = [val for val in q_id_entities for _ in (0, 1)]
    # xlabels = q_id_entities
    # ylabels = q_terms
    main_title = 'Entity Centric Matrix Input\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(m_type) + "\nlabel: " + rel_score)

    #visualize input matrix for q
    # print("input matrix for q:", q_id, q_body)
    # show_matrix(xlabels, ylabels, input_neural, title_axis_x, title_axis_y, main_title)

    #visualize output matrix of conv1 for q
    print("predit matrix of conv2 output for q:", q_id, q_body)
    input_neural = np.array([input_neural])
    input_neural = np.expand_dims(input_neural, 3) #expand input_neural to four dimension (samples_cnt, rows, columns, channel_cnt)

    model_ec_conv1 = Sequential()
    model_ec_conv1.add(Conv2D(weights=model.layers[0].get_weights(), filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu",input_shape=(14,400,1)))  # 1.2 know terms of entities importancy
    model_ec_conv1.add(MaxPooling2D(weights=model.layers[1].get_weights(), pool_size=(1, 5), strides=(1, 5)))  # 2. get max important term five by five
    model_ec_conv1.add(Conv2D(weights=model.layers[2].get_weights(), filters=64, kernel_size=(14, 4), strides=1, padding="same", activation="relu",))  # 3 know entities importancy
    # model_ec_conv1.add(MaxPooling2D(weights=model.layers[3].get_weights(),pool_size=(1, 4), strides=(1, 4)))  # 4. get entity iportancy by query phrace
    # model.add(AveragePooling2D(pool_size=(14, 1), strides=(14, 1)))  # 5 average of entity iportancy on total query

    predicted_array = model_ec_conv1.predict(input_neural)
    predict = predicted_array[0, :, :, 0]

    # print("input_neural.shape", input_neural.shape)
    # print("input_neural", input_neural)

    # print(predict.shape)
    # print(predict)
    # print(len(q_id_entities))
    # print(len(q_terms))

    main_title = 'Entity Centric Matrix Output of Conv2d_2\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(m_type) + "\nlabel: " + rel_score)


    # show_matrix(xlabels, ylabels, predict, title_axis_x, title_axis_y, main_title)
    show_matrix_simple(predict, title_axis_x, title_axis_y, main_title)


def get_visualize_ec_conv1(q_id, type, q_body, top_k_type_term = 50):
    global q_id_top_100_entities_dict
    #load moadel type centric weights
    weights_path_ec = "/home/arian/workSpaces/entityArticle/entity-attr-resources/typeRetrieval/data/reports/final_reports_27Bahman/EntityCentric/models_saved/input(cosine_detail20dim)_T5_V4_L1(100)_L2(1)_relu_drop(0)_opt(adam)_epoch(100)_bsize(128)_lr(0.0001)_lambda(0)_loss(mse)_topk(20)_regression.log_model.h5"
    model = get_model_ec()
    model.load_weights(weights_path_ec)
    print(model.summary())
    #load query terms and type terms
    q_terms = q_body.split(" ")
    type_entities_cnt = get_type_entities_cnt(type)


    #load columns, entities retrieved for this query
    m_type = type
    # q_id_top_100_entities_dict = json.load(open("./q_ret_100_entities_lm.json"))
    m_q_id_top_100_entities_dict = q_id_top_100_entities_dict[q_id]
    q_id_entities = [item[1] for item in m_q_id_top_100_entities_dict[:20]]

    q_id_entities_terms_2d = [item[6] for item in m_q_id_top_100_entities_dict[:20]]
    q_id_entities_terms = []
    for entity_terms in q_id_entities_terms_2d:
        for term, tf_idfscore in entity_terms[:20]:
            q_id_entities_terms.append(term)

        if len(entity_terms)<20:
            cnt_loop = 20 - len(entity_terms)
            for i in range(cnt_loop):
                q_id_entities_terms.append(" ")

    # q_id_entities_terms = [item[6][:20] for item in q_id_top_100_entities_dict]
    # q_id_entities_terms = np.array(q_id_entities_terms)
    # print(q_id_entities_terms)
    # print(q_id_entities)

    #load type centric matrix for this q_id and type
    train_set_entity_centric_top20_dict = json.load(open("./trainset_translation_matrix_2d_qword_tope(20topterm(20.out"))
    q_id_list_2d_matrix = train_set_entity_centric_top20_dict[q_id]
    input_neural = None
    rel_score = None
    for train_set in q_id_list_2d_matrix:
        if train_set[2] == type:
            input_neural = train_set[0]
            rel_score = train_set[1]
            # input_neural = np.array(input_neural) * (1/type_entities_cnt)
            input_neural = np.array(input_neural)
            # input_neural = np.array(input_neural)

    # print(q_id_top_100_entities_dict)






    #set detail of plot
    title_axis_x = 'entities terms(sorted by lm)'
    title_axis_y = 'query terms'
    xlabels = q_id_entities_terms[:40]
    ylabels = q_terms
    main_title = 'Entity Centric Matrix Input\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(m_type) + "\nlabel: " + rel_score)

    #visualize input matrix for q
    print("input matrix for q:", q_id, q_body)
    show_matrix(xlabels, ylabels, (input_neural[:14, :40] * 1/type_entities_cnt), title_axis_x, title_axis_y, main_title)

    #visualize output matrix of conv1 for q
    print("predit matrix of conv2 output for q:", q_id, q_body)
    input_neural = np.array([input_neural])
    input_neural = np.expand_dims(input_neural, 3) #expand input_neural to four dimension (samples_cnt, rows, columns, channel_cnt)

    model_ec_conv1 = Sequential()
    model_ec_conv1.add(Conv2D(weights=model.layers[0].get_weights(), filters=32, kernel_size=(14, 5), strides=1, padding="same", activation="relu",input_shape=(14,400,1)))  # 1.2 know terms of entities importancy

    predicted_array = model_ec_conv1.predict(input_neural)
    predict = predicted_array[0, :, :, 0]

    print("input_neural.shape", input_neural.shape)
    # print("input_neural", input_neural)

    print(predict.shape)
    # print(predict)
    print(len(q_id_entities))
    print(len(q_terms))

    main_title = 'Entity Centric Matrix Output of Conv2d_1\nquery: ' + str(' '.join(q_terms) + '\ntype:' + str(m_type) + "\nlabel: " + rel_score)


    show_matrix(xlabels, ylabels, predict[:14,:40], title_axis_x, title_axis_y, main_title)
    # show_matrix_simple(predict, title_axis_x, title_axis_y, main_title)


top_k_type_terms = 50

######################laye aval type centric##################33
# q_id = "SemSearch_LS-42"
# type_key = "<dbo:Person>"
# q_body = "twelve tribes or sons of Israel"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "SemSearch_LS-42"
# type_key = "<dbo:Album>"
# q_body = "twelve tribes or sons of Israel"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)

q_id = "QALD2_te-9"
type_key = "<dbo:MusicalWork>"
q_body = "Give me a list of all trumpet players that were bandleaders."
get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)

# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)


# q_id = "INEX_LD-2012318"
# type_key = "<dbo:Film>"
# q_body = "Directed Bela Glen Glenda Bride Monster Plan 9 Outer Space"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "SemSearch_LS-42"
# type_key = "<dbo:EthnicGroup>"
# q_body = "twelve tribes or sons of Israel"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_tr-59"
# type_key = "<dbo:Person>"
# q_body = "Give me all people with first name Jimmy"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_tr-21"
# type_key = "<dbo:AdministrativeRegion>"
# q_body = "Which states border Illinois"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# #TODO:: good example ! tavajohe bishtari be first three karde ! shayad talash konam ba run change she rankinesh!
# q_id = "QALD2_te-9"
# type_key = "<dbo:MusicalArtist>"
# q_body = "Give me a list of all trumpet players that were bandleaders"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# # khoobie ine mesal ine ke chandin javabe mohtamel dare ama  neural tuneste dar rank 2vom doros tashkhisesh bede !
# q_id = "QALD2_te-84"
# type_key = "<dbo:ComicsCreator>"
# q_body = "Who created the comic Captain America"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)

# q_id = "QALD2_tr-65"
# type_key = "<dbo:Company>"
# q_body = "Which companies work in the aerospace industry as well as\n on nuclear reactor technology"
# get_visualize_tc_conv1(q_id, type_key, q_body, top_k_type_terms)
#


######################laye dovom type centric##################33

# q_id = "INEX_LD-2012318"
# type_key = "<dbo:Film>"
# q_body = "Directed Bela Glen Glenda Bride Monster Plan 9 Outer Space"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "SemSearch_LS-42"
# type_key = "<dbo:EthnicGroup>"
# q_body = "twelve tribes or sons of Israel"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_tr-59"
# type_key = "<dbo:Person>"
# q_body = "Give me all people with first name Jimmy"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_tr-21"
# type_key = "<dbo:AdministrativeRegion>"
# q_body = "Which states border Illinois"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)
#
#
# #TODO:: good example ! tavajohe bishtari be first three karde ! shayad talash konam ba run change she rankinesh!
# q_id = "QALD2_te-9"
# type_key = "<dbo:MusicalArtist>"
# q_body = "Give me a list of all trumpet players that were bandleaders"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)
#
#
# # khoobie ine mesal ine ke chandin javabe mohtamel dare ama  neural tuneste dar rank 2vom doros tashkhisesh bede !
# q_id = "QALD2_te-84"
# type_key = "<dbo:ComicsCreator>"
# q_body = "Who created the comic Captain America"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)

# q_id = "QALD2_tr-65"
# type_key = "<dbo:Company>"
# q_body = "Which companies work in the aerospace industry as well as\n on nuclear reactor technology"
# get_visualize_tc_conv2(q_id, type_key, q_body, top_k_type_terms)




######################laye 3vom type centric##################33
#
# q_id = "INEX_LD-2012318"
# type_key = "<dbo:Film>"
# q_body = "Directed Bela Glen Glenda Bride Monster Plan 9 Outer Space"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "SemSearch_LS-42"
# type_key = "<dbo:EthnicGroup>"
# q_body = "twelve tribes or sons of Israel"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_tr-59"
# type_key = "<dbo:Person>"
# q_body = "Give me all people with first name Jimmy"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_tr-21"
# type_key = "<dbo:AdministrativeRegion>"
# q_body = "Which states border Illinois"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# #TODO:: good example ! tavajohe bishtari be first three karde ! shayad talash konam ba run change she rankinesh!
# q_id = "QALD2_te-9"
# type_key = "<dbo:MusicalArtist>"
# q_body = "Give me a list of all trumpet players that were bandleaders"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# # khoobie ine mesal ine ke chandin javabe mohtamel dare ama  neural tuneste dar rank 2vom doros tashkhisesh bede !
# q_id = "QALD2_te-84"
# type_key = "<dbo:ComicsCreator>"
# q_body = "Who created the comic Captain America"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)
#
# q_id = "QALD2_tr-65"
# type_key = "<dbo:Company>"
# q_body = "Which companies work in the aerospace industry as well as\n on nuclear reactor technology"
# get_visualize_tc_conv3(q_id, type_key, q_body, top_k_type_terms)

###################################################################################################
########################## Entity Centric #####################

######################laye aval type centric##################33

#
#
# q_id = "INEX_XER-100"
# type_key = "<dbo:Software>"
# q_body = "Operating systems to which Steve Jobs related"
# get_visualize_ec_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "INEX_LD-2012390"
# type_key = "<dbo:BaseballPlayer>"
# q_body = "baseball player most homeruns national league"
# get_visualize_ec_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
#
# q_id = "INEX_XER-94"
# type_key = "<dbo:Automobile>"
# q_body = "Hybrid cars sold in Europe"
# get_visualize_ec_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_te-28"
# type_key = "<dbo:Film>"
# q_body = "Give me all movies directed by Francis Ford Coppola"
# get_visualize_ec_conv1(q_id, type_key, q_body, top_k_type_terms)
#
#
#
# q_id = "SemSearch_ES-105"
# type_key = "<dbo:Building>"
# q_body = "cedar garden apartments"
# get_visualize_ec_conv1(q_id, type_key, q_body, top_k_type_terms)
#

##########################################################3
# q_id = "INEX_XER-100"
# type_key = "<dbo:Software>"
# q_body = "Operating systems to which Steve Jobs related"
# get_visualize_ec_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "INEX_LD-2012390"
# type_key = "<dbo:BaseballPlayer>"
# q_body = "baseball player most homeruns national league"
# get_visualize_ec_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
#
# q_id = "INEX_XER-94"
# type_key = "<dbo:Automobile>"
# q_body = "Hybrid cars sold in Europe"
# get_visualize_ec_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
# q_id = "QALD2_te-28"
# type_key = "<dbo:Film>"
# q_body = "Give me all movies directed by Francis Ford Coppola"
# get_visualize_ec_conv3(q_id, type_key, q_body, top_k_type_terms)
#
#
#
# q_id = "SemSearch_ES-105"
# type_key = "<dbo:Building>"
# q_body = "cedar garden apartments"
# get_visualize_ec_conv3(q_id, type_key, q_body, top_k_type_terms)
#



#
#
# q_id = "INEX_XER-100"
# type_key = "<dbo:Film>"
# q_body = "Operating systems to which Steve Jobs related"
# get_visualize_ec_conv3(q_id, type_key, q_body, top_k_type_terms)






sys.exit(1)
# model = get_model_ec()
# model.load_weights(weights_path_ec)

# model = get_model_tc()
# model.load_weights(weights_path_tc)

# plt.savefig("conv1_predict_input_food.png", bbox_inches='tight')

# path = "./entityCentric_Conv5X5_filters/"
# path = "./typeCentric_Conv5X5_filters/"
# save_fig_conv_filters(model, 0, path)
# save_fig_conv_filters_subplot(model, 0, path)


