# coding=utf-8

# coding: utf-8

# NOTE - need to run this in the right environment
# This is based on this code: https://github.com/codekansas/keras-language-modeling/blob/master/keras_models.py

import logging
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from keras.layers import Bidirectional
from keras.layers.core import Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.preprocessing import sequence
from numpy.random import shuffle

from IdGenerator import IdGenerator as idGen
from Rpfa import micro_rpfa
from Settings import Settings
from load_data import load_process_essays
from results_procesor import ResultsProcessor, __MICRO_F1__
from window_based_tagger_config import get_config
from wordtagginghelper import merge_dictionaries

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

DEV_SPLIT = 0.1

settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder                     = root_folder + "Training/"
test_folder                         = root_folder + "Test/"

training_pickled = settings.data_directory + "CoralBleaching/Thesis_Dataset/training.pl"
models_folder = root_folder + "Models/Bi-LSTM/"

train_config = get_config(training_folder)
processor = ResultsProcessor()

""" LOAD DATA """
train_tagged_essays = load_process_essays(**train_config)

shuffle(train_tagged_essays)

test_config = dict(train_config.items())
test_config["folder"] = test_folder

test_tagged_essays = load_process_essays(**test_config)
logger.info("Essays loaded- Train: %i Test %i" % (len(train_tagged_essays), len(test_tagged_essays)))

print("Started at: " + str(datetime.now()))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

from numpy.random import shuffle

# ## Prepare Tags
tag_freq = defaultdict(int)
unique_words = set()
for essay in train_tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            unique_words.add(word)
            for tag in tags:
                tag_freq[tag] += 1

EMPTY_TAG = "Empty"
regular_tags = list((t for t in tag_freq.keys() if t[0].isdigit()))
vtags = set(regular_tags)
vtags.add(EMPTY_TAG)

ix2tag = {}
for ix, t in enumerate(vtags):
    ix2tag[ix] = t

generator = idGen(seed=1)  # important as we zero pad sequences

maxlen = 0
for essay in train_tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            id = generator.get_id(word)  # starts at 0, but 0 used to pad sequences
        maxlen = max(maxlen, len(sentence) + 2)

# test essays may be longer
for essay in test_tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            id = generator.get_id(word)  # starts at 0, but 0 used to pad sequences
        maxlen = max(maxlen, len(sentence) + 2)

def ids2tags(ids):
    return [generator.get_key(j) for j in ids]

def lbls2tags(ixs):
    return [ix2tag[ix] for ix in ixs]

START = "<start>"
END = "<end>"

def get_training_data(tessays):
    # outputs
    xs = []
    ys = []
    ys_bytag = defaultdict(list)
    seq_lens = []

    # cut texts after this number of words (among top max_features most common words)
    for essay in tessays:
        for sentence in essay.sentences:
            row = []
            y_found = False
            y_seq = []
            for word, tags in [(START, set())] + sentence + [(END, set())]:
                id = generator.get_id(word)  # starts at 0, but 0 used to pad sequences
                row.append(id)

                # remove unwanted tags
                tags = vtags.intersection(tags)
                # retain all tags for evaluation (not just most common)
                # SKIP the START and END tags
                if word != START and word != END:
                    for t in (vtags - set([EMPTY_TAG])):
                        if t in tags:
                            ys_bytag[t].append(1)
                        else:
                            ys_bytag[t].append(0)

                # encode ys with most common tag only
                if len(tags) > 1:
                    most_common = max(tags, key=lambda t: tag_freq[t])
                    tags = set([most_common])
                if len(tags) == 0:
                    tags.add(EMPTY_TAG)

                one_hot = []
                for t in vtags:
                    if t in tags:
                        one_hot.append(1)
                    else:
                        one_hot.append(0)
                y_seq.append(one_hot)

            seq_lens.append(len(row) - 2)
            ys.append(y_seq)
            xs.append(row)

    xs = sequence.pad_sequences(xs, maxlen=maxlen)
    ys = sequence.pad_sequences(ys, maxlen=maxlen)
    assert xs.shape[0] == ys.shape[0], "Sequences should have the same number of rows"
    assert xs.shape[1] == ys.shape[1] == maxlen, "Sequences should have the same lengths"
    return xs, ys, ys_bytag, seq_lens

# ## Create Train - Test Split
# Helper Functions
def collapse_results(seq_lens, preds):
    assert len(seq_lens) == preds.shape[0], "Axis 1 size does not align"
    pred_ys_by_tag = defaultdict(list)
    for i in range(len(seq_lens)):
        row_ixs = preds[i, :]
        len_of_sequence = seq_lens[i] + 2
        # sequences are padded from the left, take the preds from the end of the seq
        pred_ys = [ix2tag[j] for j in row_ixs[-len_of_sequence:]]
        # skip the start and end label
        pred_ys = pred_ys[1:-1]
        for pred_tag in pred_ys:
            pred_ys_by_tag[pred_tag].append(1)
            # for all other tags, a 0
            for tag in (vtags - set([EMPTY_TAG, pred_tag])):
                pred_ys_by_tag[tag].append(0)
        if EMPTY_TAG in pred_ys_by_tag:
            del pred_ys_by_tag[EMPTY_TAG]
    return pred_ys_by_tag

def train_dev_split(lst, dev_split):
    # random shuffle
    shuffle(lst)
    num_training = int((1.0 - dev_split) * len(lst))
    return lst[:num_training], lst[num_training:]


folds = [(train_tagged_essays, test_tagged_essays)]

fold2training_data = {}
fold2dev_data = {}
fold2test_data = {}

for i, (essays_TD, essays_VD) in enumerate(folds):
    # further split into train and dev test
    essays_train, essays_dev = train_dev_split(essays_TD, DEV_SPLIT)
    fold2training_data[i] = get_training_data(essays_train)
    fold2dev_data[i]      = get_training_data(essays_dev)
    # Test Data
    fold2test_data[i]     = get_training_data(essays_VD)

# ## Load Glove 100 Dim Embeddings

# see /Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/DeepLearning/WordVectors/pickle_glove_embedding.py
# for creating pre-filtered embeddings file

embeddings_file = "/Users/simon.hughes/data/word_embeddings/glove.6B/cb_dict_glove.6B.100d.txt"
# read data file
with open(embeddings_file, "rb+") as f:
    cb_emb_index = pickle.load(f)

missed = set()
for wd in unique_words:
    if wd not in cb_emb_index:
        missed.add(wd)

# ### Construct Embedding Matrix

EMBEDDING_DIM = list(cb_emb_index.values())[0].shape[0]

def get_embedding_matrix(words, idgenerator, max_features, init='uniform', unit_length=False):
    embedding_dim = list(cb_emb_index.values())[0].shape[0]
    # initialize with a uniform distribution
    if init == 'uniform':
        # NOTE: the max norms for these is quite low relative to the embeddings
        embedding_matrix = np.random.uniform(low=-0.05, high=0.05, size=(max_features, embedding_dim))
    elif init == 'zeros':
        embedding_matrix = np.zeros(shape=(max_features, embedding_dim), dtype=np.float32)
    elif init == 'normal':
        raise Exception("Need to compute the mean and sd")
        #embedding_matrix = np.random.normal(mean, sd, size=(max_features, embedding_dim))
    else:
        raise Exception("Unknown init type")
    for word in words:
        i = idgenerator.get_id(word)
        embedding_vector = cb_emb_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    if unit_length:
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        # remove 0 norms to prevent divide by zero
        norms[norms == 0.0] = 1.0
        embedding_matrix = embedding_matrix / norms
    return embedding_matrix

def score_predictions(model, xs, ys_by_tag, seq_len):
    preds = model.predict_classes(xs, batch_size=batch_size, verbose=0)
    pred_ys_by_tag = collapse_results(seq_len, preds)
    class2metrics = ResultsProcessor.compute_metrics(ys_by_tag, pred_ys_by_tag)
    micro_metrics = micro_rpfa(class2metrics.values())
    return micro_metrics, pred_ys_by_tag

def get_predictions(model, xs, ys_by_tag, seq_len):
    preds = model.predict_classes(xs, batch_size=batch_size, verbose=0)
    pred_ys_by_tag = collapse_results(seq_len, preds)
    return pred_ys_by_tag

def get_ts():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def get_file_ts():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

embedding_size = EMBEDDING_DIM
hidden_size = 128
out_size = len(vtags)
batch_size = 128

# ## Train Bi-Directional LSTM With Glove Embeddings
max_features = len(generator.get_ids()) + 2  # Need plus one maybe due to masking of sequences

# merge_mode is Bi-Directional only
def evaluate_fold(fold_ix, use_pretrained_embedding, bi_directional, num_rnns, merge_mode, hidden_size):
    if use_pretrained_embedding:
        embedding_matrix = get_embedding_matrix(unique_words, generator, max_features, init='uniform',
                                                unit_length=False)
        embedding_layer = Embedding(max_features,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=True,
                                    mask_zero=True)  # If false, initialize unfound words with all 0's
    else:
        embedding_layer = Embedding(max_features, embedding_size, input_length=maxlen, trainable=True, mask_zero=True)

    if bi_directional:
        rnn_layer_fact = lambda: Bidirectional(GRU(hidden_size, return_sequences=True, consume_less="cpu"),
                                               merge_mode=merge_mode)
    else:
        rnn_layer_fact = lambda: GRU(hidden_size, return_sequences=True, consume_less="cpu")

    model = Sequential()
    model.add(embedding_layer)
    for i in range(num_rnns):
        model.add(rnn_layer_fact())

    model.add(TimeDistributedDense(out_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode="temporal")

    X_train, y_train, train_ys_by_tag, seq_len_train = fold2training_data[fold_ix]
    X_dev, y_dev, dev_ys_by_tag, seq_len_dev = fold2dev_data[fold_ix]
    X_test, y_test, test_ys_by_tag, seq_len_test = fold2test_data[fold_ix]

    # init loop vars
    f1_scores = [-1]
    num_since_best_score = 0
    patience = 3
    best_weights = None

    for i in range(30):
        print("{ts}: Epoch={epoch}".format(ts=get_ts(), epoch=i))
        epochs = 1  # epochs per training instance
        results = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.0, verbose=0)
        micro_metrics, _ = score_predictions(model, X_dev, dev_ys_by_tag, seq_len_dev)
        print(micro_metrics)

        f1_score = micro_metrics.f1_score
        best_f1_score = max(f1_scores)
        if f1_score <= best_f1_score:
            num_since_best_score += 1
        else:  # score improved
            num_since_best_score = 0
            best_weights = model.get_weights()

        f1_scores.append(f1_score)
        if num_since_best_score >= patience:
            break

    # load best weights
    model.set_weights(best_weights)
    train_predictions_by_tag = get_predictions(model, X_train, train_ys_by_tag, seq_len_train)
    test_predictions_by_tag = get_predictions(model, X_test, test_ys_by_tag, seq_len_test)
    return train_predictions_by_tag, test_predictions_by_tag, train_ys_by_tag, test_ys_by_tag

# ## Hyper Param Tuning
def cross_validation(use_pretrained_embedding, bi_directional, num_rnns, merge_mode, hidden_size):
    results = Parallel(n_jobs=1)(
        delayed(evaluate_fold)(i, use_pretrained_embedding, bi_directional, num_rnns, merge_mode, hidden_size) for i in
        range(len(fold2training_data)))

    cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)
    for result in results:
        td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag = result
        merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
        merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    SUFFIX = "_RNN_MOST_COMMON_TAG"
    CB_TAGGING_TD, CB_TAGGING_VD = "TEST_CB_TAGGING_TD" + SUFFIX, "TEST_CB_TAGGING_VD" + SUFFIX
    parameters = dict(train_config)
    parameters["extractors"] = []
    parameters["min_feat_freq"] = 0

    parameters["use_pretrained_embedding"] = use_pretrained_embedding
    parameters["bi-directional"] = bi_directional
    parameters["hidden_size"] = hidden_size
    parameters["merge_mode"] = merge_mode
    parameters["num_rnns"] = num_rnns

    wd_algo = "RNN"
    wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag,
                                               parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag,
                                               parameters, wd_algo)
    avg_f1 = float(processor.get_metric(CB_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1


import warnings
warnings.filterwarnings("ignore")

i = 0
for use_pretrained_embedding in [True]:
    for bi_directional in [True]:
        for num_rnns in [2]:
            for merge_mode in ["sum"]:
                for hidden_size in [256]:
                    i += 1
                    print("[{i}] Params {ts} - Embeddings={use_pretrained_embedding}, Bi-Direct={bi_directional} Num_Rnns={num_rnns} Hidden_Size={hidden_size}".format(
                        i=i, ts=get_ts(), use_pretrained_embedding=use_pretrained_embedding,
                        bi_directional=bi_directional, num_rnns=num_rnns, hidden_size=hidden_size))

                    micro_f1 = cross_validation(use_pretrained_embedding, bi_directional, num_rnns, merge_mode, hidden_size)
                    print("MicroF1={micro_f1}".format(micro_f1=micro_f1))




