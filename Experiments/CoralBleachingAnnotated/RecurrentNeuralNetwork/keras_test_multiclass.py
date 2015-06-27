from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from collections import defaultdict
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1
import keras.layers.convolutional

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

#TARGET_Y            = "Causer"
TARGET_Y            = "14"

TEST_SPLIT          = 0.2
PRE_PEND_PREV_SENT  = 0 #2 seems good
REVERSE             = False
PREPEND_REVERSE     = False

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

generator = idGen()
xs = []
ys = []
END_TAG = 'END'

# cut texts after this number of words (among top max_features most common words)
maxlen = 0

tag_freq = defaultdict(int)
for essay in tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            for tag in tags:
                if (tag[-1].isdigit() or tag in {"Causer", "explicit", "Result"} \
                        or tag.startswith("Causer") or tag.startswith("Result") or tag.startswith("explicit"))\
                        and not ("Anaphor" in tag or "rhetorical" in tag or "other" in tag or "->" in tag):
                    tag_freq[tag] += 1

freq_tags = set((tag for tag, freq in tag_freq.items() if freq >= 20))

lst_freq_tags = sorted(freq_tags)
ix2tag = {}
for i, tag in enumerate(lst_freq_tags):
    ix2tag[i] = tag


from numpy.random import shuffle
shuffle(tagged_essays)

for essay in tagged_essays:
    sent_rows = [[generator.get_id(END_TAG)] for i in range(PRE_PEND_PREV_SENT)]
    for sentence in essay.sentences:
        row = []

        un_tags = set()
        for word, tags in sentence + [(END_TAG, set())]:
            id = generator.get_id(word)
            row.append(id)
            for tag in tags:
                un_tags.add(tag)

        y = []
        for tag in lst_freq_tags:
            y.append(1 if tag in un_tags else 0)

        sent_rows.append(row)
        ys.append(y)

        if PRE_PEND_PREV_SENT > 0:
            x = []
            for i in range(PRE_PEND_PREV_SENT + 1):
                ix = i+1
                if ix > len(sent_rows):
                    break
                x = sent_rows[-ix] + x
            row = x

        if PREPEND_REVERSE:
            row = row[::-1] + row
        xs.append(row)
        maxlen = max(len(xs[-1]), maxlen)
max_features=generator.max_id() + 2
batch_size = 16

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))

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
#model.add(LSTM(embedding_size, 128)) # try using a GRU instead, for fun
#model.add(GRU(embedding_size, embedding_size)) # try using a GRU instead, for fun
model.add(JZS1(embedding_size, 64)) # try using a GRU instead, for fun
#JSZ1, embedding = 64, 64 hidden = 0.708
#model.add(Dropout(0.2))
model.add(Dense(64, len(lst_freq_tags)))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

# Does very well (F1 0.684) using embedding 0.64 and hidden = 0.64

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

def rnd(v):
    digits = 6
    return str(round(v, digits)).ljust(digits+2)

# convert to numpy array for slicing
y_train, y_test = np.asarray(y_train), np.asarray(y_test)

def test(epochs = 1):
    results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, show_accuracy=True, verbose=1)
    probs = model.predict_proba(X_test, batch_size=batch_size)
    f1s = []
    for ix, tag in ix2tag.items():
        tag_predictions = probs[:, ix]
        tag_ys = y_test[:, ix]
        r, p, f1, cutoff = find_cutoff(tag_ys, tag_predictions)
        print(tag.ljust(35), str(sum(tag_ys)).ljust(3), "recall", rnd(r), "precision", rnd(p), "f1", rnd(f1), "cutoff",
              rnd(cutoff))
        f1s.append(f1)
    mean_f1 = np.mean(f1s)
    print("MEAN F1: " + str(mean_f1))
    return mean_f1
    return f1

best = 0
while True:
    iterations += 1

    accuracy = test(1)
    best = max(best, accuracy)
    if accuracy < last_accuracy:
        decreases +=1
    else:
        decreases = 0

    print("Best F1: ", best)
    if decreases >= 30 and iterations > 10:
        print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy

#results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=  5, validation_split=0.2, show_accuracy=True, verbose=1)
print("at: " + str(datetime.datetime.now()))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam
