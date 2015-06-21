from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge, RepeatVector
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

print('Build model...')
# input: 2D tensor of integer indices of characters (eg. 1-57).
# input tensor has shape (samples, maxlen)
nb_feature_maps = 32
embedding_size = 64

ngram_filters = [3, 5, 7]
conv_filters = []

for n_gram in ngram_filters:
    sequential = Sequential()
    conv_filters.append(sequential)
    #sequential = Embedding(max_features, embedding_size)
    #sequential.add(Reshape(1, maxlen, embedding_size))
    sequential.add(Convolution2D(nb_feature_maps, 1, n_gram, max_features))
    sequential.add(Activation("relu"))
    sequential.add(MaxPooling2D(poolsize=(maxlen - n_gram + 1, 1)))
    sequential.add(Flatten())

model = Sequential()
#model.add(RepeatVector(len(ngram_filters)))
model.add(Merge(conv_filters, mode='sum'))
model.add(Dense(nb_feature_maps, 1))
model.add(Activation("sigmoid"))

#model.add(Dropout(0.25))
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

    if decreases >= 4 and iterations > 10:
        print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy

#results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=  5, validation_split=0.2, show_accuracy=True, verbose=1)
print("at: " + str(datetime.datetime.now()))

# Causer: recall 0.746835443038 precision 0.670454545455 f1 0.706586826347 - 32 embedding, lstm, sigmoid, adam

""" Raises Error:

Started at: 2015-06-20 16:50:22.107033
Results Dir: /Users/simon.hughes/Google Drive/Phd/Results/
Data Dir:    /Users/simon.hughes/Google Drive/Phd/Data/
Root Dir:    /Users/simon.hughes/GitHub/NlpResearch/
Public Data: /Users/simon.hughes/GitHub/NlpResearch/Data/PublicDatasets/
Loading data...
Pad sequences (samples x time)
XS Shape:  (2084, 1, 85, 883)
1667 train sequences
417 test sequences
X_train shape: (1667, 1, 85, 883)
X_test shape: (417, 1, 85, 883)
Build model...
Train...
Epoch 0
Traceback (most recent call last):
  File "/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingAnnotated/ConvolutionalNeuralNetwork/keras_merged_convnet_test.py", line 152, in <module>
    accuracy = test(1)
  File "/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/CoralBleachingAnnotated/ConvolutionalNeuralNetwork/keras_merged_convnet_test.py", line 143, in test
    results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, show_accuracy=True, verbose=1)
  File "build/bdist.macosx-10.6-x86_64/egg/keras/models.py", line 204, in fit
  File "/Users/simon.hughes/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Theano-0.6.0-py2.7.egg/theano/compile/function_module.py", line 497, in __call__
    allow_downcast=s.allow_downcast)
  File "/Users/simon.hughes/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/Theano-0.6.0-py2.7.egg/theano/tensor/type.py", line 157, in filter
    data.shape))
TypeError: ('Bad input argument to theano function at index 1(0-based)', 'Wrong number of dimensions: expected 4, got 2 with shape (16, 1).')

Process finished with exit code 1
"""