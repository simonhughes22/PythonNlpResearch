from Metrics import rpf1
from Decorators import memoize_to_disk
from load_data import load_process_essays
from IterableFP import flatten

from window_based_tagger_config import get_config
from IdGenerator import IdGenerator as idGen
# END Classifiers

import Settings
import logging

import datetime
print("Started at: " + str(datetime.datetime.now()))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

#TARGET_Y            = "explicit"
TARGET_Y            = "Causer"
#TARGET_Y            = "14"
TEST_SPLIT          = 0.2
PRE_PEND_PREV_SENT  = 0 #2 seems good for some. Perfect score on 14 with pre-pend=0
REVERSE             = False
PREPEND_REVERSE     = False
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"

config = get_config(folder)
config["stem"] = True
config["lower_case"] = False

""" FEATURE EXTRACTION """
""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )

generator = idGen()
xs = []
ys = []
END_TAG = 'END'

from numpy.random import shuffle
shuffle(tagged_essays)

# cut texts after this number of words (among top max_features most common words)
maxlen = 0
for essay in tagged_essays:

    sent_rows = [[generator.get_id(END_TAG)] for i in range(PRE_PEND_PREV_SENT)]
    for sentence in essay.sentences:
        row = []
        y_found = False
        for word, tags in sentence + [(END_TAG, set())]:
            id = generator.get_id(word)
            row.append(id)
            if TARGET_Y in tags:
                y_found = True

        sent_rows.append(row)
        ys.append(1 if y_found else 0)

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

from passage.layers import Embedding
from passage.layers import GatedRecurrent
from passage.layers import LstmRecurrent
from passage.layers import Dense

from passage.models import RNN
from passage.utils import save, load

print("Loading data...")
num_training = int((1.0 - TEST_SPLIT) * len(xs))

X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]

num_feats = generator.max_id() + 1

layers = [
    Embedding(size=64, n_features=num_feats),
    #LstmRecurrent(size=32),
    #NOTE - to use a deep RNN, you need all but the final layers with seq_ouput=True
    #GatedRecurrent(size=64, seq_output=True),
    GatedRecurrent(size=64, direction= 'backward' if REVERSE else 'forward'),
    #LstmRecurrent(size=128),
    Dense(size=1, activation='sigmoid'),
]

#emd 64, gru 64 is good - 0.70833 causer (0 prev sents)

print("Creating Model")
model = RNN(layers=layers, cost='bce')

def find_cutoff(y_test, predictions):
    scale = 100.0

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

def test(epochs=1):
    model.fit(X_train, y_train, n_epochs=epochs, batch_size=64)#64 seems good for now
    predictions = flatten(model.predict(X_test))
    r, p, f1, cutoff = find_cutoff(y_test, predictions)
    print("recall", rnd(r), "precision", rnd(p), "f1", rnd(f1), "cutoff", rnd(cutoff))
    return f1

last_f1 = -1
decreases = 0
max_f1 = 0.0

while True:
    f1 = test(1)
    if f1 < last_f1:
        decreases += 1
    max_f1 = max(f1, max_f1)
    if decreases > 10:
        print "Stopping, f1 %f is less than previous f1 %f" % (f1, last_f1)
        break
    last_f1 = f1

print "Max:", rnd(max_f1)
bp = 0
#save(model, 'save_test.pkl')
#model = load('save_test.pkl')

""" This model, although doing sequential prediction, predicts a tag per document not per word. """