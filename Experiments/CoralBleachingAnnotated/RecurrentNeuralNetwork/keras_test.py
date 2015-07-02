from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1
import keras.layers.convolutional
from keras.constraints import maxnorm


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
from IterableFP import flatten
# END Classifiers

import Settings
import logging

import datetime
print("Started at: " + str(datetime.datetime.now()))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

TARGET_Y            = "Causer"
#TARGET_Y            = "14"

TEST_SPLIT          = 0.2
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"

config = get_config(folder)
config["stem"] = True

""" FEATURE EXTRACTION """
""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )

from numpy.random import shuffle
shuffle(tagged_essays)

generator = idGen()
generator.get_id("......")
xs = []
ys = []

# cut texts after this number of words (among top max_features most common words)
maxlen = 0
for essay in tagged_essays:
    for sentence in essay.sentences:
        row = []
        y_found = False
        for word, tags in sentence:
            id = generator.get_id(word) + 1 #starts at 0, but 0 used to pad sequences
            row.append(id)
            if TARGET_Y in tags:
                y_found = True
        ys.append(1 if y_found else 0)
        xs.append(row)
        maxlen = max(len(row), maxlen)

max_features=generator.max_id() + 2
batch_size = 16

print("Loading data...")
num_training = int((1.0 - TEST_SPLIT) * len(xs))

X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")

MAX_LEN = maxlen
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN) #30 seems good
X_test  = sequence.pad_sequences(X_test,  maxlen=MAX_LEN)

#def reverse(lst):
#    return lst[::-1]
#X_train, X_test = np.asarray( map(reverse, X_train) ), np.asarray( map(reverse, X_test))

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

embedding_size = 64

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size))

model.add(Reshape((1, vector_len)))

#model.add(LSTM(embedding_size, 64)) # try using a GRU instead, for fun
#model.add(GRU(embedding_size, embedding_size)) # try using a GRU instead, for fun
model.add(JZS1(embedding_size, 64)) # try using a GRU instead, for fun
#JSZ1, embedding = 64, 64 hidden = 0.708
#model.add(Dropout(0.2))
model.add(Dense(64, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#model.compile(loss='hinge', optimizer='adagrad', class_mode="binary")

print("Train...")
last_accuracy = 0
iterations = 0
decreases = 0

def find_cutoff(y_test, predictions):
    scale = 20.0

    min_val = round(min(predictions))
    max_val = round(max(predictions))
    diff = max_val - min_val
    inc = diff / scale

    cutoff = -1
    best = -1
    for i in range(1, int(scale)+1, 1):
        val = inc * i
        classes = [1 if p >= val else 0 for p in predictions]
        r, p, f1 = rpf1(y_test, classes)
        if f1 >= best:
            cutoff = val
            best = f1

    classes = [1 if p >= cutoff else 0 for p in predictions]
    r, p, f1 = rpf1(y_test, classes)
    return r, p, f1, cutoff

def test(epochs = 1):
    results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, show_accuracy=True, verbose=1)
    probs = flatten( model.predict_proba(X_test, batch_size=batch_size) )
    model.predict()
    r, p, f1, cutoff = find_cutoff(y_test, probs)
    print("recall", r, "precision", p, "f1", f1, "cutoff", cutoff)
    return f1

while True:
    iterations += 1

    accuracy = test(1)
    if accuracy < last_accuracy:
        decreases +=1
    else:
        decreases = 0

    if decreases >= 2 and iterations > 10:
        print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy

#results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=  5, validation_split=0.2, show_accuracy=True, verbose=1)
print("at: " + str(datetime.datetime.now()))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam
