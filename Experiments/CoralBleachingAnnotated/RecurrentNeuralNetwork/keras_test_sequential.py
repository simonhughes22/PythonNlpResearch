from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1

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

MIN_WORD_FREQ       = 5        # 5 best so far
TARGET_Y            = "Causer"
#TARGET_Y            = "14"
TEST_SPLIT          = 0.2
SEQ                 = True
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"

config = get_config(folder)

""" FEATURE EXTRACTION """
""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )

generator = idGen()
xs = []
ys = []
ys_seq = []

# cut texts after this number of words (among top max_features most common words)
maxlen = 0
for essay in tagged_essays:
    for sentence in essay.sentences:
        row = []
        y_found = False
        y_seq = []
        for word, tags in sentence:
            id = generator.get_id(word) + 1 #starts at 0, but 0 used to pad sequences
            row.append(id)
            if TARGET_Y in tags:
                y_found = True
                y_seq.append(1)
            else:
                y_seq.append(0)
        ys.append(1 if y_found else 0)
        ys_seq.append(y_seq)
        xs.append(row)
        maxlen = max(len(row), maxlen)

max_features=generator.max_id() + 2
batch_size = 128

def transform_outputs(ys):
    return [map(lambda c: [c], row) for row in ys]

def collapse_results(ys):
    return [row.max() for row in ys]

def collapse_probs(probs, threshold = 0.0):
    return [1.0 if max(row) >= threshold else 0.0 for row in probs]

def reverse(xs):
    return [x[::-1] for x in xs]

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))

X_train, y_reg_train, yseq_train, X_test, y_reg_test, yseq_test = xs[:num_training], ys[:num_training], ys_seq[:num_training], xs[num_training:], ys[num_training:], ys_seq[num_training:]

y_train = yseq_train
y_test  = yseq_test

#X_train, X_test, y_train, y_test = reverse(X_train), reverse(X_test), reverse(y_train), reverse(y_test)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")

MAX_LEN = maxlen
X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN) #30 seems good
X_test  = sequence.pad_sequences(X_test,  maxlen=MAX_LEN)

y_train = sequence.pad_sequences(y_train, maxlen=MAX_LEN)
y_test  = sequence.pad_sequences(y_test,  maxlen=MAX_LEN)

y_train = np.asarray(transform_outputs(y_train))
y_test  = np.asarray(transform_outputs(y_test))

print('X_train shape:', X_train.shape)
print('X_test shape:',  X_test.shape)

print('y_train shape:', y_train.shape)
print('y_test shape:',  y_test.shape)

""" REVERSE """

embedding_size = 64
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, embedding_size))

model.add(GRU(embedding_size, 64, return_sequences=True)) # try using a GRU instead, for fun
model.add(TimeDistributedDense(64, 1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
last_accuracy = 99999
iterations = 0
decreases = 0

best = None
prev_weights = None

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

def train(epochs=1):
    results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.2, show_accuracy=True, verbose=1)

    probs = model.predict_proba(X_test, batch_size=batch_size)
    predictions = collapse_results(probs)

    r, p, f1, cutoff = find_cutoff(y_reg_test, predictions)
    print("recall", r, "precision", p, "f1", f1, "cutoff", cutoff)
    return f1

while True:
    iterations += 1
    accuracy = train(1)

    if accuracy < last_accuracy:
        best = prev_weights
        decreases +=1
    else:
        decreases = 0

    if decreases >= 10 and iterations > 30:
        print("Val accruacy decreased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy
    prev_weights = model.get_output()

print("at: " + str(datetime.datetime.now()))
print("Best F1:", best)

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam


