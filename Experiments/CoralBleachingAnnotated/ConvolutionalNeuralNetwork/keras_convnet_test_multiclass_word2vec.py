from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import keras.layers.convolutional
from keras.constraints import maxnorm

from Metrics import rpf1

from Decorators import memoize_to_disk
from load_data import load_process_essays

from window_based_tagger_config import get_config
from IdGenerator import IdGenerator as idGen
from IterableFP import flatten
import Word2Vec_load_vectors
from numpy import random

get_vector = Word2Vec_load_vectors.fact_get_vector()

# END Classifiers

import Settings
import logging

import datetime
print("Started at: " + str(datetime.datetime.now()))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

TEST_SPLIT          = 0.2
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"

config = get_config(folder)
config["min_df"] = 1
config["stem"] = False
# Note: See below. I store the regular and lc versions of each word
config["lower_case"] = True

""" FEATURE EXTRACTION """
""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
random.shuffle(tagged_essays)

regular_tags = set()
for essay in tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            for tag in tags:
                if tag.isdigit() or tag in {"Causer", "explicit", "Result"}:
                    regular_tags.add(tag)

def rnd(v):
    digits = 6
    return str(round(v, digits)).ljust(digits + 2)

lst_regular_tags = sorted(regular_tags)
ix2tag = {}
for i, tag in enumerate(lst_regular_tags):
    ix2tag[i] = tag

generator = idGen()
xs = []
ys = []

# cut texts after this number of words (among top max_features most common words)
maxlen = 0
for essay in tagged_essays:
    for sentence in essay.sentences:
        row = []
        un_tags = set()
        for word, tags in sentence:
            vec = get_vector(word)
            row.append(vec)
            for tag in tags:
                un_tags.add(tag)
        y = []
        for tag in lst_regular_tags:
            y.append(1 if tag in un_tags else 0)

        xs.append(row)
        maxlen = max(len(row), maxlen)
        ys.append(y)

batch_size = 16

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))

print("Pad sequences (samples x time)")
MAX_LEN = maxlen

# LEFT pad with zero vectors
emb_shape = xs[0][0].shape
print("Embedding Shape:", emb_shape)
for row in xs:
    while len(row) < MAX_LEN:
        zeros = np.zeros(emb_shape)
        row.insert(0, zeros)

xs = np.asarray(xs)
xs = xs.reshape((xs.shape[0], 1, xs.shape[1], xs.shape[2]))
print("XS Shape: ", xs.shape)

ys = np.asarray(ys)

X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
# input: 2D tensor of integer indices of characters (eg. 1-57).
# input tensor has shape (samples, maxlen)
nb_feature_maps = 64
embedding_size = emb_shape[0]

conv_filters = []
ngram_filters = [3, 4, 5, 7]
for n_gram in ngram_filters:
    sequential = Sequential()
    conv_filters.append(sequential)

    sequential.add(Convolution2D(nb_feature_maps, 1, n_gram, embedding_size))
    sequential.add(Activation("relu"))
    sequential.add(MaxPooling2D(poolsize=(MAX_LEN - n_gram + 1, 1)))
    sequential.add(Flatten())

model = Sequential()
model.add(Merge(conv_filters, mode='concat'))
model.add(Dropout(0.5))
model.add(Dense(nb_feature_maps * len(conv_filters), len(regular_tags)))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
last_accuracy = 0
iterations = 0
decreases = 0
best = -1

concat_X_test  = []
for i in range(len(ngram_filters)):
    concat_X_test.append(X_test)

def test(epochs = 1):
    ixs = range(len(X_train))
    random.shuffle(ixs)
    x_shf = X_train[ixs]
    y_shf = y_train[ixs]

    concat_X_train = []
    for i in range(len(ngram_filters)):
        concat_X_train.append(x_shf)

    results = model.fit(concat_X_train, y_shf, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, show_accuracy=True, verbose=1)
    predictions = model.predict_proba(concat_X_test)
    print("Xp shape:", predictions.shape)
    f1s = []
    for ix, tag in ix2tag.items():
        tag_predictions = predictions[:, ix]
        tag_predictions = [1 if p >= 0.5 else 0 for p in tag_predictions]
        tag_ys = y_test[:, ix]
        r, p, f1 = rpf1(tag_ys, tag_predictions)
        count = sum(tag_ys)
        print(tag.ljust(10), str(count).rjust(4), "recall", rnd(r), "precision", rnd(p), "f1", rnd(f1))
        f1s.append(f1)
    mean_f1 = np.mean(f1s)
    print("MEAN F1: " + str(mean_f1))
    return mean_f1

while True:
    iterations += 1

    print("Iteration:", str(iterations))
    accuracy = test(1)
    if accuracy < last_accuracy:
        decreases +=1
    else:
        decreases = 0
    best = max(best, accuracy)

    #if decreases >= 4 and iterations > 10:
    #    print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
    #    break
    last_accuracy = accuracy

#results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=  5, validation_split=0.2, show_accuracy=True, verbose=1)
print("at: " + str(datetime.datetime.now()))
print("Best:" + str(best))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam