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
            id = generator.get_id(word)
            row.append(id)
            if TARGET_Y in tags:
                y_found = True
        ys.append(1 if y_found else 0)
        xs.append(row)
        maxlen = max(len(row), maxlen)

from passage.layers import Embedding
from passage.layers import GatedRecurrent
from passage.layers import LstmRecurrent
from passage.layers import Dense

from passage.models import RNN
from passage.utils import save, load

print("Loading data...")
num_training = int((1.0 - 0.2) * len(xs))
X_train, y_train, X_test, y_test = xs[:num_training], ys[:num_training], xs[num_training:], ys[num_training:]

num_feats = generator.max_id() + 1

layers = [
    Embedding(size=128, n_features=num_feats),
    #LstmRecurrent(size=32),
    GatedRecurrent(size=32),
    Dense(size=1, activation='sigmoid'),
]

#emd 128, gru 32 is good - 0.70006 causer

print("Creating Model")
model = RNN(layers=layers, cost='bce')

last_f1 = -1
decreases = 0

def test(epochs=1):
    model.fit(X_train, y_train, n_epochs=epochs, batch_size=8)#8 seems good for now
    predictions = flatten(model.predict(X_test))
    if min(predictions) < 0.0:
        mid_point = 0.0
    else:
        mid_point = 0.5

    classes = [1 if p >= mid_point else 0 for p in predictions]
    r, p, f1 = rpf1(y_test, classes)
    print("recall", r, "precision", p, "f1", f1)
    return f1

while True:
    f1 = test(1)
    if f1 < last_f1:
        decreases += 1

    if decreases > 3:
        print "Stopping, f1 %f is less than previous f1 %f" % (f1, last_f1)
        break
    last_f1 = f1

bp = 0
#save(model, 'save_test.pkl')
#model = load('save_test.pkl')

""" This model, although doing sequential prediction, predicts a tag per document not per word. """