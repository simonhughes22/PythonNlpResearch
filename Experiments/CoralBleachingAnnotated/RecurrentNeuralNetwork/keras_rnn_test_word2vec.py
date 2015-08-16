from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
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

MIN_WORD_FREQ       = 5        # 5 best so far
TARGET_Y            = "Causer"
#TARGET_Y            = "14"

TEST_SPLIT          = 0.2
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
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

generator = idGen()
xs = []
ys = []

# cut texts after this number of words (among top max_features most common words)
maxlen = 0
for essay in tagged_essays:
    for sentence in essay.sentences:
        row = []
        y_found = False
        for word, tags in sentence:
            vec = get_vector(word)
            row.append(vec)
            if TARGET_Y in tags:
                y_found = True
        ys.append(1 if y_found else 0)
        xs.append(row)
        maxlen = max(len(row), maxlen)

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
#Not needed for recurrent
#xs = xs.reshape((xs.shape[0], 1, xs.shape[1], xs.shape[2]))
print("XS Shape: ", xs.shape)

X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
# input: 2D tensor of integer indices of characters (eg. 1-57).
# input tensor has shape (samples, maxlen)
nb_feature_maps = 32
n_ngram = 5 # 5 is good (0.7338 on Causer) - 64 sized embedding, 32 feature maps, relu in conv layer
embedding_size = emb_shape[0]
hidden_size = 64

model = Sequential()
model.add(GRU(embedding_size, hidden_size))
model.add(Dense(hidden_size, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#model.compile(loss='hinge', optimizer='adagrad', class_mode="binary")

print("Train...")
last_accuracy = 0
iterations = 0
decreases = 0
best = -1

def test(epochs = 1):
    results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, show_accuracy=True, verbose=1)
    classes = flatten( model.predict_classes(X_test, batch_size=batch_size) )
    r, p, f1 = rpf1(y_test, classes)
    print("recall", r, "precision", p, "f1", f1)
    return f1

while True:
    iterations += 1

    accuracy = test(1)
    if accuracy < last_accuracy:
        decreases +=1
    else:
        decreases = 0
    best = max(best, accuracy)

    if decreases >= 4 and iterations > 10:
        print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy

#results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=  5, validation_split=0.2, show_accuracy=True, verbose=1)
print("at: " + str(datetime.datetime.now()))
print("Best:" + str(best))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam