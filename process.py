import os
import nltk
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from joblib import dump, load

nltk.download('wordnet')
TRAIN_DIR = 'data/train-set'
TEST_DIR = 'data/test-set'
# TRAIN_DIR = 'ling-spam/train-mails'
# TEST_DIR = 'ling-spam/test-mails'

MAX_WORDS = 3000
N_FILES = 0

def return_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory)])

def normalise_dict(dictionary):
    list_to_remove = list(dictionary.keys())
    for item in list_to_remove:
        if item.isalpha() == False or len(item) == 1:
            del dictionary[item]

def create_dict(train_dir):
    all_words = []
    files = return_files(train_dir)
    for mail in files:
        with open(mail) as m:
            words = nltk.word_tokenize(m.read())
            all_words += words
    dictionary = Counter(all_words)
    normalise_dict(dictionary)
    dictionary = dictionary.most_common(MAX_WORDS)
    dump(dictionary, 'dictionary.joblib')

def extract_features(directory):
    dictionary = load('dictionary.joblib')
    files = return_files(directory)
    features_matrix = np.zeros((len(files), MAX_WORDS))
    for i, fil in enumerate(files):
        with open(fil) as text:
            words = text.read().split()
            for word in words:
                for k, word_count in enumerate(dictionary):
                    if word_count[0] == word:
                        features_matrix[i, k] = words.count(word)
    return features_matrix

def extract_text_features(file):
    dictionary = load('dictionary.joblib')
    features_matrix = np.zeros((1, MAX_WORDS))
    with open(file) as text:
        words = nltk.word_tokenize(text.read())
        for word in words:
            for k, word_count in enumerate(dictionary):
                if word_count[0] == word:
                    features_matrix[0, k] = words.count(word)
    return features_matrix
