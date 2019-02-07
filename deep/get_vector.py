import os,sys
dirname = os.path.dirname(__file__)
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from scipy import spatial
import numpy as np
from nltk.corpus import wordnet as wn
"""
sudo python3.6 -m deep.get_vector "architectural structure musuem building university diocese bridge roman architecture"
"""
word2vec_train_set_path = os.path.join(dirname, '../data/GoogleNews-vectors-negative300.bin')
word_vectors = None

def loadWord2Vec():
    global word_vectors
    if word_vectors is None:
        print("w2v loading...")
        word_vectors = KeyedVectors.load_word2vec_format(word2vec_train_set_path, binary=True)
        print("w2v loaded...")

def getVector(word):
    loadWord2Vec()
    if word in word_vectors:
        vector = word_vectors[word]
        return vector
    else:
        # print(word + "\t not in w2v google news vocab !")
        return []


def get_cosine_similarity(q_w2v_word, entity_avg_w2v):
    cosine_sim = 1 - spatial.distance.cosine(q_w2v_word, entity_avg_w2v)
    return cosine_sim

def get_average_w2v(tokens):
    token_resume = 0

    vector = []
    np_array = None

    # print("tokens: ", tokens)
    # print("\n\n")
    # print("tokens len: ", len(tokens))

    while ((len(vector) == 0)):
        if (token_resume == len(tokens)):
            break

        first_token_exist_in_w2v = tokens[token_resume]
        # print("token_resume: ", token_resume)

        vector = getVector(first_token_exist_in_w2v)

        if len(vector) > 0:
            np_array = np.array([np.array(vector)])

        token_resume += 1
    if len(vector) == 0:
        print("tamame token haye query dar w2v nabudand ! :(, tokens:", tokens)
        return np.zeros(300)  # kare ghalati vali vase inke ta akhar run ejra beshe felan!

    for token in tokens[token_resume:]:
        # print('token ', token)
        vector = getVector(token)
        if len(vector) > 0:
            tmp_array = np.array(vector)
            np_array = np.concatenate((np_array, [tmp_array]))

    vector_mn = np_array.mean(axis=0)  # to take the mean of each row, 300D vec
    return vector_mn

def get_similiar(tokens):
	loadWord2Vec()
	for token in tokens:
		# word_vectors.similar_by_vector(word_vectors["survey"], topn=1)
		return str(word_vectors.most_similar(positive=[token], topn=1))


terms = sys.argv[1].split(" ")
print(terms)

for term in terms:
	vec1 = getVector(term)
	# print(term," vec len", len(vec1), "similar words: ", get_similiar([term]))
	print(term," vec len", len(vec1))
	for term2 in terms:
		vec2 = getVector(term2)
		cos_sim = get_cosine_similarity(vec1, vec2)
		print("\tcos(",term,",",term2,")=", str(cos_sim))


q1 = "roman architecture" #sys.argv[2].split(",")
# types =  ["Architectural Structure", "architectural structure", 'musuem', 'Musuem', 'building', 'Building', 'university', 'University', ""] #sys.argv[3].split(",")
types =  ["architectural structure", 'musuem', 'building', 'university', "diocese", "bridge"] #sys.argv[3].split(",")
############
for t in types:
	print("\n\n\n")
	tokens_q = q1.split(" ")
	token_t = t.split(" ")
	q_avg = get_average_w2v(tokens_q)
	t_avg = get_average_w2v(token_t)

	cos_sim = get_cosine_similarity(q_avg, t_avg)
	print("cos(",q1,",",t,")=", str(cos_sim))
##########