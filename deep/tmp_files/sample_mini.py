import os, json, random, sys
from keras.layers import *
from termcolor import colored


import numpy as np
import keras_metrics
import sys

from keras.layers.core import Dense, Activation
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.layers import *
from keras.models import *


import os, subprocess
import json,ast
import ast, sys, re, random, csv, json, math

from deep import train_set_generator as tsg
# import utils.file_utils as file_utils
# import utils.list_utils as list_utils

import numpy as np
import pandas as pd

# from scipy import spatial
#


# sentences = [
# 	q1 = "what is the best mountain",
# 	d1_p = "everest is the good mountain",
# 	d1_n = "lake of khazar",
# ]
# print(sen//tences)
# sys.exit(1)
q1 = "what is the best mountain"
d1_p = "everest is the good mountain"
d1_n = "lake of khazar"

q2 = "degree of albert einstein"
d2_p = "fiziks in sweden"
d2_n = "ride bycicle with bus in afternoon"


q3 = "what is the good mountain"
d3_p = "everest is the best mountain"
d3_n = "lake of khalij"

def get_ngram(text, n):
	return [text[i:i+n] for i in range(len(text)-1)]

def get_vector(text):
	sentences = []
	sentences.append(q1)
	sentences.append(d1_p)
	sentences.append(d1_n)
	sentences.append(q2)
	sentences.append(d2_p)
	sentences.append(d2_n)


	sentences.append(q3)
	sentences.append(d3_p)
	sentences.append(d3_n)

	sentences.sort()


	all_texts = (list([" ".join(i.split(" ")) for i in sentences]))
	all_texts = list(set(all_texts))
	all_texts = [i.split(" ") for i in all_texts]
	all_words = []
	for sentence in all_texts:
		for word in sentence:
			all_words.append(word)

	all_words = list(set(all_words))

	all_words = ["#"+i+"#" for i in all_words]
	all_words = list([get_ngram(i,3) for i in all_words])
	all_words.sort()
	all_words = list(all_words)

	all_grams = []
	for ngrams in all_words:
		for gram in ngrams:
			all_grams.append(gram)
	all_grams.sort()
	all_grams = list(set(all_grams))

	all_grams_dict = dict()
	cn = -1
	for gram in all_grams:
		cn +=1
		all_grams_dict[str(gram)] = cn

	
	text = text.split(" ")
	text = ["#"+i+"#" for i in text]
	text = list([get_ngram(i,3) for i in text])
	text = list(text)
	
	text_to_ngram_vec = []
	vec_len = len(all_grams)


	for i in range(vec_len):
		text_to_ngram_vec.append(0)
	
	text_grams = []
	for ngrams in text:
		for gram in ngrams:
			text_grams.append(gram)


	for w in text_grams:
		index_w = all_grams_dict[w]
		text_to_ngram_vec[index_w] += 1
	
	# print("text ngram vector space len ", len(all_grams_dict))
	# print("text_to_ngram_vec len ", len(text_to_ngram_vec))
	# print("text_to_ngram vector ",text_to_ngram_vec)
	return text_to_ngram_vec


vec_q1 = get_vector(q1)
# print("input text : " , q1)
# print("vector output:", vec_q1)
len_vec = len(vec_q1)
print("len_vec: " + str(len_vec))

model_query = Sequential()
#layer 1
model_query.add(Dense(300, input_shape=(len_vec,)))
model_query.add(Activation('tanh'))
#layer 2
model_query.add(Dense(300))
model_query.add(Activation('tanh'))
#layer 3
model_query.add(Dense(128))
model_query.add(Activation('tanh'))


#for relative doc in this implementation
model_doc_1 = Sequential()
#layer 1
model_doc_1.add(Dense(300, input_shape=(len_vec,)))
model_doc_1.add(Activation('tanh'))
#layer 2
model_doc_1.add(Dense(300))
model_doc_1.add(Activation('tanh'))
#layer 3
model_doc_1.add(Dense(128))
model_doc_1.add(Activation('tanh'))


merged_q1_d1 = Sequential()

merged_q1_d1 = Add()([model_query.output, model_doc_1.output])
merged_q1_d1 = Dense(1) (merged_q1_d1)
merged_q1_d1 = Activation('tanh') (merged_q1_d1)

merged_q1_d1_newModel = Sequential()
merged_q1_d1_newModel = Model([model_query.input,model_doc_1.input], merged_q1_d1)


merged_q1_d1_newModel.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
# train_X = [[get_vector(q1),get_vector(q1), get_vector(q2), get_vector(q2)], [get_vector(d1_p),get_vector(d1_n), get_vector(d2_p), get_vector(d2_n)]]
# train_Y = [1, 0, 1, 0]

# merged_q1_d1_newModel.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
train_X = [[get_vector(q1), get_vector(q1)], [get_vector(d1_p),get_vector(d1_n)]]
train_Y = [1, 0]
merged_q1_d1_newModel.fit(train_X, train_Y, epochs=100, batch_size=1)


test_X = [[get_vector(q3), get_vector(q3)], [get_vector(d3_p), get_vector(d3_n)]]
test_Y = [1, 0]

print(merged_q1_d1_newModel.evaluate(test_X, test_Y, verbose=0))
# sys.exit(1)
loss, accuracy, precision, recall = merged_q1_d1_newModel.evaluate(test_X, test_Y, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
print("precision = {:.2f}".format(accuracy))
print("recall = {:.2f}".format(accuracy))
print("loss = {:.2f}".format(loss))