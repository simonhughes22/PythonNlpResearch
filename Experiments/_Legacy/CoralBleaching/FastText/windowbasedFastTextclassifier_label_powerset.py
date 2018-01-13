import logging
import os
from collections import defaultdict
from random import randint

from nltk.tag.crf import CRFTagger

import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from load_data import load_process_essays
from nltk_datahelper import to_label_powerset_tagged_sentences
from nltk_datahelper import to_sentences, to_flattened_binary_tags_by_code
from results_procesor import ResultsProcessor
from tag_frequency import get_tag_freq, regular_tag
from window_based_tagger_config import get_config
from wordtagginghelper import merge_dictionaries

from NgramGenerator import compute_ngrams

import fasttext

SENT_START  = "@@@"
SENT_END    = "###"

def window_to_sequence(window):
    i = 0
    sequenced = []
    for token in window:
        sequenced.append(str(i).rjust(2,"X") + ":" + token)
        i += 1
    return sequenced

def tagged_sents_to_word_windows(tagged_sents, window_size):

    offset = int((window_size-1) / 2)
    tagged_windows = []
    for sent in tagged_sents:
        wds, tags = zip(*sent)
        wds = list(wds)
        # pad sentence
        for _ in range(offset):
            wds.insert(0, SENT_START)
            wds.append(SENT_END)

        windows = compute_ngrams(wds, max_len=window_size, min_len=window_size)
        #numbered_windows = map(window_to_sequence, windows)
        #tagged = zip(numbered_windows, tags)
        tagged = zip(windows, tags)
        tagged_windows.extend(tagged)

    return tagged_windows

def tagged_word_window_to_fastext_format(tagged_window, lbl = "__label__"):
    tokens, tag = tagged_window
    return "{lbl}{tag} , {sent}".format(lbl=lbl, tag=tag, sent=" ".join(tokens))

def tagged_windows_to_file(tagged_windows, fname):
    with open(fname, "w+") as f:
        for tag_win in tagged_windows:
            f.write(tagged_word_window_to_fastext_format(tag_win) + "\n")

from collections import defaultdict

def predictions_to_ys_by_code(predicted_powerset_lbls, expected_tags):

    expected_tags = set(expected_tags)
    ys_by_code = defaultdict(list)

    for lst_lbls in predicted_powerset_lbls:
        # the predicted labels themselves are a list, so join also, then split the whole shebang
        lbl = ",".join(lst_lbls)
        tags = set(lbl.split(","))
        for expected in expected_tags:
            ys_by_code[expected].append(1 if expected in tags else 0)
    return ys_by_code

def train_classifer_on_fold(essays_TD, essays_VD, regular_tags, fold, window_size):

    # Start Training
    print("Fold %i Training code" % fold)

    # For training
    td_sents = to_label_powerset_tagged_sentences(essays_TD, regular_tags)
    vd_sents = to_label_powerset_tagged_sentences(essays_VD, regular_tags)

    # To word windows
    td_tagged_windows = tagged_sents_to_word_windows(td_sents, window_size)
    vd_tagged_windows = tagged_sents_to_word_windows(vd_sents, window_size)

    model_filename    = "{folder}/model_{fold}_{random}".format(folder=models_folder, fold=fold, random=randint(0, 9999999))
    training_filename = "{folder}/training_{fold}_{random}.txt".format(folder=models_folder, fold=fold, random=randint(0, 9999999))

    tagged_windows_to_file(td_tagged_windows, training_filename)

    # TRAIN MODEL
    model = fasttext.supervised(training_filename, model_filename)

    td_predictions = model.predict( map( lambda (tokens, tag): " ".join(tokens), td_tagged_windows))
    vd_predictions = model.predict( map( lambda (tokens, tag): " ".join(tokens), vd_tagged_windows))

    # for evaluation - binary tags
    # YS (ACTUAL)
    wd_td_ys_bytag = to_flattened_binary_tags_by_code(td_sents, regular_tags)
    wd_vd_ys_bytag = to_flattened_binary_tags_by_code(vd_sents, regular_tags)

    # YS (PREDICTED)
    td_wd_predictions_by_code = predictions_to_ys_by_code(td_predictions, regular_tags)
    vd_wd_predictions_by_code = predictions_to_ys_by_code(vd_predictions, regular_tags)

    #os.remove(model_filename)
    #os.remove(training_filename)

    return wd_td_ys_bytag, wd_vd_ys_bytag, td_wd_predictions_by_code, vd_wd_predictions_by_code

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Load the Essays
# ---------------
# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS = True

MIN_FEAT_FREQ = 5  # 5 best so far
CV_FOLDS = 5

MIN_TAG_FREQ = 5
LOOK_BACK = 0  # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()
folder = settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
models_folder = settings.data_directory + "CoralBleaching/models/FastText"

config = get_config(folder)
print(config)

mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays(**config)[:100]
logger.info("Essays loaded")
len(tagged_essays)

# Create Corpus in CRF Format (list of list of tuples(word,tag))
# --------------------------------------------------------------

tag_freq = get_tag_freq(tagged_essays)
freq_tags = list(set((tag for tag, freq in tag_freq.items() if freq >= MIN_TAG_FREQ and regular_tag(tag))))
regular_tags = [t for t in freq_tags if t[0].isdigit()]

""" FEATURE EXTRACTION """
config["window_size"] = 11
offset = (config["window_size"] - 1) / 2


cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)
folds = cross_validation(tagged_essays, CV_FOLDS)

results = [train_classifer_on_fold(essays_TD, essays_VD, regular_tags, fold, config["window_size"])
                for fold, (essays_TD, essays_VD) in enumerate(folds)]

for result in results:
    wd_td_ys_bytag, wd_vd_ys_bytag, td_wd_predictions_by_code, vd_wd_predictions_by_code = result

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

logger.info("Training completed")

""" Persist Results to Mongo DB """
wd_algo = "FASTTEXT_LBL_POWERSET"
SUFFIX = "_FASTTEXT_LBL_POWERSET"
CB_TAGGING_TD, CB_TAGGING_VD= "CB_TAGGING_TD" + SUFFIX, "CB_TAGGING_VD" + SUFFIX

parameters = dict(config)
parameters["extractors"] = ["None"]
parameters["min_feat_freq"] = MIN_FEAT_FREQ

wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

# This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
print processor.results_to_string(wd_td_objectid, CB_TAGGING_TD, wd_vd_objectid, CB_TAGGING_VD, "TAGGING")
logger.info("Results Processed")

