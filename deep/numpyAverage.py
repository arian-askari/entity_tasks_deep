
import numpy as np
import sys

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models.keyedvectors import KeyedVectors

# arrayt = np.array([[1,2,3],[3,6,9]])
arrayt = np.array([[1,8,2]])
arrayt = np.concatenate((arrayt, [[1,2,3]])) #append
arrayt = np.concatenate((arrayt, [[3,6,9]])) #append
arrayt = np.concatenate((arrayt, [[11,7,3]])) #append
mn = arrayt.mean(axis=1)     # to take the mean of each row
mn = arrayt.mean(axis=0)     # to take the mean of each col
print(mn)