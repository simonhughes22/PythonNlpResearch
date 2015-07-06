from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import keras.layers.convolutional
import theano.tensor as T

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
from numpy import random
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
#config["min_df"] = 5

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

def rnd(v):
    digits = 6
    return str(round(v, digits)).ljust(digits+2)

max_features=generator.max_id() + 2
batch_size = 4

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))

print("Pad sequences (samples x time)")
MAX_LEN = maxlen

xs = sequence.pad_sequences(xs, maxlen=MAX_LEN) #30 seems good

print("XS Shape: ", xs.shape)

X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


print('X_train shape:', X_train.shape)
print('X_test shape:',  X_test.shape)

y_train, y_test = np.asarray(y_train), np.asarray(y_test)

print('Y_train shape:', y_train.shape)
print('Y_test shape:',  y_test.shape)

print('Build model...')
# input: 2D tensor of integer indices of characters (eg. 1-57).
# input tensor has shape (samples, maxlen)

nb_feature_maps = 128
embedding_size = 64

graph = Graph()
graph.add_input(name="input", dtype='int')
#graph.inputs["input"].input = T.imatrix()
graph.add_node(Embedding(max_features, embedding_size), name="embedding", input="input")
graph.add_node(Reshape(1, MAX_LEN, embedding_size), name="reshape", input="embedding")

top_layer_names = []
ngram_filters = [3, 4, 5, 7]
for n_gram in ngram_filters:
    sngram = str(n_gram)
    conv_layer = 'conv-' + sngram
    activ_layer = 'activ-' + sngram
    pool_layer = 'max-pool-' + sngram
    flat_layer = "flat-" + sngram
    dropout_layer = "dropout-" + sngram
    top_layer_names.append(dropout_layer)

    graph.add_node(Convolution2D(nb_feature_maps, 1, n_gram, embedding_size), name=conv_layer, input='reshape')
    graph.add_node(Activation("relu"), name=activ_layer, input=conv_layer)
    graph.add_node(MaxPooling2D(poolsize=(MAX_LEN - n_gram + 1, 1)), name=pool_layer, input=activ_layer)
    graph.add_node(Flatten(), name=flat_layer, input=pool_layer)
    graph.add_node(Dropout(0.5), name=dropout_layer, input=flat_layer)

graph.add_node(Dense(nb_feature_maps * len(ngram_filters), len(regular_tags)), name="dense1", inputs=top_layer_names, merge_mode='concat')
graph.add_node(Activation("sigmoid"), name="activ-top", input='dense1')
graph.add_output(name='output', input='activ-top')

graph.compile('adam', {"output" : "mse"})
model = graph

print("Train...")
last_accuracy = 0
iterations = 0
decreases = 0
best = 0

#Xp = model.predict(X_train)
#print("Xp shape:", Xp.shape)

def test(epochs=1):

    ixs = range(len(X_train))
    random.shuffle(ixs)
    x_shf = X_train[ixs]
    y_shf = y_train[ixs]

    model.fit({"input": X_train, "output": y_train}, nb_epoch=epochs)#64 seems good for now
    predictions = model.predict({"input": X_test, "output": y_test})["output"]
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

    accuracy = test(3)
    if accuracy < last_accuracy:
        decreases +=1
    else:
        decreases = 0
    best = max(best, accuracy)

    if decreases >= 15 and iterations > 10:
        print("Val Loss increased from %f to %f. Stopping" % (last_accuracy, accuracy))
        break
    last_accuracy = accuracy

print("at: " + str(datetime.datetime.now()))
print("Best:" + str(best))
