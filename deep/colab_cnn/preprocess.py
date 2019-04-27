from sklearn.preprocessing import StandardScaler
import numpy as np
import re

def get_normalize2D(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return np.array(scaler.transform(data))

def is_exist_whole_word(word, text, case_sensitive=True):
    if case_sensitive == True:
        with_case = re.compile(r'\b({0})\b'.format(word))
        return len(with_case.findall(text))>0

    else:
        word = word.lower()
        text = text.lower()
        with_case = re.compile(r'\b({0})\b'.format(word))
        return len(with_case.findall(text))>0

# print(is_exist_whole_word("arian", "Arian askari ariana")) #True
# print(is_exist_whole_word("arian", "Arian askari ariana", case_sensitive=False)) #True
