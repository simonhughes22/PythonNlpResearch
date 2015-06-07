from pybrain.datasets import SequentialDataSet
from itertools import cycle
import numpy as np

from Decorators import memoize_to_disk
from load_data import load_process_essays

from window_based_tagger_config import get_config
from IdGenerator import IdGenerator as idGen
# END Classifiers

import Settings
import logging

import datetime
from Metrics import rpf1
print("Started at: " + str(datetime.datetime.now()))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

MIN_WORD_FREQ       = 5        # 5 best so far
#TARGET_Y            = "Causer"
TARGET_Y            = "Causer"
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

train_ds = SequentialDataSet(max_features, 1)

for xrow, yrow in zip(xs, ys_seq):
    train_ds.newSequence()
    for x,y in zip(xrow, yrow):
        one_hot = [0] * max_features
        one_hot[x] = 1
        train_ds.addSample(one_hot, y)

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer

net = buildNetwork(max_features, 32, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

from pybrain.supervised import RPropMinusTrainer

trainer = RPropMinusTrainer(net, dataset=train_ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 5
CYCLES = 100
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)
    train_errors.append(trainer.testOnData())
    epoch = (i+1) * EPOCHS_PER_CYCLE
    print "/n epoch %i/%i" % (epoch, EPOCHS)

print()
print("final error =", train_errors[-1])