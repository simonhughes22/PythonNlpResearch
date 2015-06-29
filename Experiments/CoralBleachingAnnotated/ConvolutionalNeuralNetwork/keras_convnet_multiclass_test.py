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

from Metrics import rpf1

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

MIN_WORD_FREQ       = 5        # 5 best so far
TARGET_Y            = "Causer"
#TARGET_Y            = "14"

TEST_SPLIT          = 0.2
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

regular_tags = set()
for essay in tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            for tag in tags:
                if tag.isdigit() or tag in {"Causer", "explicit", "Result"}:
                    regular_tags.add(tag)

lst_regular_tags = sorted(regular_tags)
ix2tag = {}
for i, tag in enumerate(lst_regular_tags):
    ix2tag[i] = tag

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

            #NOTE - put a space in when using characters
            un_tags = set()
            for word, tags in sentence:
                id = generator.get_id(word)
                row.append(id)
                for tag in tags:
                    un_tags.add(tag)
        y = []
        for tag in lst_regular_tags:
            y.append(1 if tag in un_tags else 0)

        ys.append(y)
        xs.append(row)
        maxlen = max(len(row), maxlen)

max_features=generator.max_id() + 2
batch_size = 4

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))

print("Pad sequences (samples x time)")
MAX_LEN = maxlen

xs = sequence.pad_sequences(xs, maxlen=MAX_LEN) #30 seems good

def get_one_hot(id):
    zeros = [0] * max_features
    zeros[id] = 1
    return zeros

#new_xs = []
#for x in xs:
#    new_x = [get_one_hot(id) for id in x ]
#    new_xs.append(new_x)

#xs = np.asarray(new_xs)
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
nb_feature_maps = 8
n_ngram = 4 # 5 is good (0.7338 on Causer) - 64 sized embedding, 32 feature maps, relu in conv layer
embedding_size = 64

model = Sequential()
model.add(Embedding(max_features, embedding_size))
model.add(Reshape(1, maxlen, embedding_size))
model.add(Convolution2D(nb_feature_maps, 1, n_ngram, embedding_size))
model.add(Activation("relu"))
#model.add(Convolution2D(nb_feature_maps, nb_feature_maps, n_ngram, embedding_size))
#model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(maxlen - n_ngram + 1, 1)))
model.add(Flatten())
model.add(Dense(nb_feature_maps, len(lst_regular_tags)))
model.add(Activation("sigmoid"))
#model.add(Dense(nb_feature_maps/2, 1))
#model.add(Activation("sigmoid"))

#model.add(Dropout(0.25))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#model.compile(loss='hinge', optimizer='adagrad', class_mode="binary")

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

print("Train...")
last_accuracy = 0
iterations = 0
decreases = 0
best = -1

y_train, y_test = np.asarray(y_train), np.asarray(y_test)
def test(epochs=1):
    model.fit(X_train, y_train, nb_epoch=epochs, batch_size=64)#64 seems good for now
    predictions = model.predict_proba(X_test)
    f1s = []
    for ix, tag in ix2tag.items():
        tag_predictions = predictions[:, ix]
        tag_ys = y_test[:, ix]
        r, p, f1, cutoff = find_cutoff(tag_ys, tag_predictions)
        print(tag.ljust(10), "recall", rnd(r), "precision", rnd(p), "f1", rnd(f1), "cutoff", rnd(cutoff))
        f1s.append(f1)
    mean_f1 = np.mean(f1s)
    print("MEAN F1: " + str(mean_f1))
    return mean_f1

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
