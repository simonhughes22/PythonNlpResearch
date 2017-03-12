# coding=utf-8
import os
import numpy as np
import pickle

GLOVE_DIR = "/Users/simon.hughes/data/word_embeddings/glove.6B"
DATA_SET = "CoralBleaching"
GLOVE_EMBEDDINGS_FILE = 'glove.6B.100d.txt'
OUTPUT_FILE = "/Users/simon.hughes/data/word_embeddings/glove.6B/cb_dict_" + GLOVE_EMBEDDINGS_FILE

"""
LOAD ESSAY FILES
"""

from load_data import load_process_essays
from window_based_tagger_config import get_config
from Settings import Settings

settings = Settings()
root_folder = settings.data_directory + DATA_SET + "/Thesis_Dataset/"
training_folder = root_folder + "Training" + "/"
config = get_config(training_folder)

tagged_essays = load_process_essays(**config)

"""
LOAD WORD VECTORS
"""

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_EMBEDDINGS_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

unique_words = set()
for essay in tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            unique_words.add(word)

cbe_matrix = {}
for wd in unique_words:
    if wd in embeddings_index:
        coeff = embeddings_index[wd]
        cbe_matrix[wd] = coeff

with open(OUTPUT_FILE, "wb+") as f:
    pickle.dump(cbe_matrix, f, fix_imports=True)

print("Embeddings saved to %s" % OUTPUT_FILE)