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
            for c in word:
                id = generator.get_id(c) + 1 #starts at 0, but 0 used to pad sequences
                row.append(id)
            if TARGET_Y in tags:
                y_found = True
        ys.append(1 if y_found else 0)
        xs.append(row)
        maxlen = max(len(row), maxlen)

max_features=generator.max_id() + 2
batch_size = 16

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))

print("Pad sequences (samples x time)")
MAX_LEN = maxlen

xs = sequence.pad_sequences(xs, maxlen=MAX_LEN) #30 seems good

def get_one_hot(id):
    zeros = [0] * max_features
    zeros[id] = 1
    return zeros

new_xs = []
for x in xs:
    new_x = [get_one_hot(id) for id in x ]
    new_xs.append(new_x)

xs = np.asarray(new_xs)
xs = xs.reshape((xs.shape[0], 1, xs.shape[1], xs.shape[2]))
print("XS Shape: ", xs.shape)

X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

embedding_size = 32

print('Build model...')
model = Sequential()

# input: 2D tensor of integer indices of characters (eg. 1-57).
# input tensor has shape (samples, maxlen)
model = Sequential()
#model.add(Embedding(max_features, 256)) # embed into dense 3D float tensor (samples, maxlen, 256)
#model.add(Reshape(1, maxlen, 256)) # reshape into 4D tensor (samples, 1, maxlen, 256)
# VGG-like convolution stack
#model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))
#model.add(Activation('relu'))
#model.add(Convolution2D(32, 1, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(poolsize=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(32,1))
#model.add(Activation("sigmoid"))

nb_feature_maps = 32
n_ngram = 20
model.add(Convolution2D(nb_feature_maps, 1, n_ngram, max_features))
model.add(MaxPooling2D(poolsize=(maxlen - n_ngram + 1, 1)))
model.add(Flatten())
model.add(Dense(nb_feature_maps, 1))
model.add(Activation("sigmoid"))
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
#model.compile(loss='hinge', optimizer='adagrad', class_mode="binary")

print("Train...")
last_accuracy = 0
iterations = 0
decreases = 0

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

    if decreases >= 2 and iterations > 10:
        print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy

#results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=  5, validation_split=0.2, show_accuracy=True, verbose=1)
print("at: " + str(datetime.datetime.now()))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam
