from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from collections import defaultdict
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
import keras.layers.convolutional
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, RepeatVector, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1
from IterableFP import flatten

from Metrics import rpf1

'''
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes:

    - RNNs are tricky. Choice of batch size is important,
    choice of loss and optimizer is critical, etc.
    Most configurations won't converge.

    - LSTM loss decrease during training can be quite different
    from what you see with CNNs/MLPs/etc. It's more or less a sigmoid
    instead of an inverse exponential.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

    250s/epoch on GPU (GT 650M), vs. 400s/epoch on CPU (2.4Ghz Core i7).
'''
from Decorators import memoize_to_disk
from load_data import load_process_essays

from window_based_tagger_config import get_config
from IdGenerator import IdGenerator as idGen
# END Classifiers

import Settings
import logging

import datetime
print("Started at: " + str(datetime.datetime.now()))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

TEST_SPLIT          = 0.0
PRE_PEND_PREV_SENT  = 0 #2 seems good
REVERSE             = False

# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"

config = get_config(folder)
config["stem"] = True

""" FEATURE EXTRACTION """
""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )

generator = idGen()
generator.get_id("......")
xs = []
ys = []

SENT_START_TAG = 'SENT_START'
SENT_END_TAG   = 'SENT_END'

ESSAY_START_TAG = 'ESSAY_START'
ESSAY_END_TAG   = 'ESSAY_END'

# cut texts after this number of words (among top max_features most common words)
maxlen = 0

from numpy.random import shuffle
shuffle(tagged_essays)

for essay in tagged_essays:
    essay_rows = [[generator.get_id(ESSAY_START_TAG)]]
    for sentence in essay.sentences:
        row = []

        un_tags = set()
        for word, tags in [(SENT_START_TAG, set())] + sentence + [(SENT_END_TAG, set())]:
            id = generator.get_id(word)
            row.append(id)
            for tag in tags:
                un_tags.add(tag)

        xs.append(essay_rows[-1])
        ys.append(row)
        essay_rows.append(row)

        maxlen = max(len(xs[-1]), maxlen)
    xs.append(essay_rows[-1])
    ys.append([generator.get_id(ESSAY_END_TAG)])
    maxlen = max(len(xs[-1]), maxlen)

max_features=generator.max_id() + 2
batch_size = 16

print("Loading data...")
num_training = int((1.0 - TEST_SPLIT) * len(xs))
num_left = len(xs) - num_training
#num_valid = int(num_left / 2.0)
num_valid = 0
num_test = len(xs) - num_training - num_valid

#MAX_LEN = maxlen
MAX_LEN = 30
print("Pad sequences (samples x time)")

def repeat_vector(vector):
    output = []
    ix = 0
    for i in range(MAX_LEN):
        if ix >= len(vector):
            ix = 0
        output.append(vector[ix])
        ix += 1
    return output

def to_one_hot(id):
    zeros = [0] * max_features
    zeros[id] = 1
    return zeros

# Reverse inputs
xs = map(lambda x: x[::-1], xs)
xs = sequence.pad_sequences(xs, maxlen=MAX_LEN)
xs = np.asarray(xs)

# don't zero pad - just predicts 0's all the time
ys = map(repeat_vector, ys)
ys = map(lambda y: map(to_one_hot, y), ys)
ys = np.asarray(ys)

"""X_train, y_train, X_valid, y_valid, X_test, y_test = \
    xs[:num_training], ys[:num_training],  \
    xs[num_training:num_training + num_valid], ys[num_training:num_training + num_valid],  \
    xs[num_training + num_valid:], ys[num_training + num_valid:]
"""

X_train, y_train = xs, ys

print(X_train.shape, 'train sequences')
print("YS Shape: ", ys.shape)

embedding_size = 32
hidden_size = 32

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(JZS1(embedding_size, hidden_size, return_sequences=True)) # try using a GRU instead, for fun
model.add(TimeDistributedDense(hidden_size, max_features, activation="softmax"))

# try using different optimizers and different optimizer configs
model.compile(loss='mse', optimizer='adam')

outp = model.predict(X_train)
print("Output Shape:", outp.shape)

print("Train...")

def ids_to_words(vector):
    s = ""
    for id in vector:
        if id == 0:
            continue
        word = generator.get_key(id)
        if word in {".", "!", "?"}:
            s += word
        else:
            s += " " + word
    return s.strip()

def max_probs_to_words(vector):
    ixs = np.argmax(vector, axis=1)
    return ids_to_words(flatten(ixs))

def test(epochs = 1):
    results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, show_accuracy=True, verbose=1)

    r = range(len(X_train))
    shuffle(r)
    ixs = r

    x_sub = X_train[ixs]
    y_sub = y_train[ixs]

    p_sub = model.predict_proba(x_sub, batch_size=batch_size)

    cnt = 0
    for x, y, p in zip(x_sub, y_sub, p_sub):
        if p.max() > 0 and cnt < 10:
            x_rev = x[::-1]
            print("X   :", ids_to_words(x_rev))
            print("Y   :", max_probs_to_words(y))
            print("Pred:", max_probs_to_words(p))
            cnt += 1

iterations = 0
while True:
    iterations += 1
    print("Iteration:", iterations)
    test(5)

print("at: " + str(datetime.datetime.now()))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam
