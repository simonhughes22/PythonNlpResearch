# coding=utf-8
from random import randint

from Decorators import memoize_to_disk
from FindFiles import find_files
from sent_feats_for_stacking import *
import dill, os
from load_data import load_process_essays, extract_features

from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from collections import defaultdict, Counter
from window_based_tagger_config import get_config
from perceptron_tagger_multiclass_combo import PerceptronTaggerMultiClassCombo
from results_procesor import ResultsProcessor, __MICRO_F1__
# END Classifiers

import Settings
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags

# Combo tags
#NOTE: this essentially forces it to ignore lbl powersets
TAG_FREQ_THRESHOLD   = 5

# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
folder =                            root_folder + "Training/"
processed_essay_filename_prefix =   root_folder + "Pickled/essays_proc_pickled_"
features_filename_prefix =          root_folder + "Pickled/feats_pickled_"
tmp_folder =                        root_folder + "tmp/"

config = get_config(folder)

""" FEATURE EXTRACTION """
config["window_size"] = 9
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed  = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed  = fact_extract_ngram_features_stemmed(offset, 2)
triigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 3)
unigram_bow_window      = fact_extract_bow_ngram_features(offset, 1)

#optimal CB feature set
extractors = [
    unigram_window_stemmed,
    biigram_window_stemmed,
    triigram_window_stemmed,

    unigram_bow_window,

    extract_dependency_relation,
    extract_brown_cluster
]

# For mongo
extractor_names = map(lambda fn: fn.func_name, extractors)
print("Extractors\n\t" + "\n\t".join(extractor_names))

feat_config = dict(config.items() + [("extractors", extractors)])

""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
logger.info("Essays loaded")

# most params below exist ONLY for the purposes of the hashing to and from disk
essay_feats = extract_features(tagged_essays, **feat_config)
logger.info("Features loaded")

""" DEFINE TAGS """
_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
all_regular_tags = list((t for t in flatten(lst_all_tags) if t[0].isdigit()))

tag_freq = Counter(all_regular_tags )
regular_tags = list(tag_freq.keys())

""" works best with all the pair-wise causal relation codes """
wd_train_tags = regular_tags
wd_test_tags  = regular_tags
# tags to evaluate against

""" CLASSIFIERS """
folds = cross_validation(essay_feats, CV_FOLDS)
def pad_str(val):
    return str(val).ljust(20) + "  "

def toDict(obj):
    return obj.__dict__

def evaluate_tagger(num_iterations, tag_history, k_fold_data_fnames):

    cv_wd_td_predictions_by_tag, cv_wd_vd_predictions_by_tag    = defaultdict(list), defaultdict(list)
    for kfold, data_filename in k_fold_data_fnames.items():

        with open(data_filename, "rb") as f:
            k_fold_data = dill.load(f)

        essays_VD, essays_VD, essays_TD_most_freq, wd_td_ys_bytag, wd_vd_ys_bytag = k_fold_data

        """ TRAINING """
        tagger = PerceptronTaggerMultiClassCombo(wd_train_tags, tag_history=tag_history,
                                                 combo_freq_threshold=TAG_FREQ_THRESHOLD)
        tagger.train(essays_TD_most_freq, nr_iter=num_iterations, verbose=False)

        """ PREDICT """
        td_wd_predictions_by_code = tagger.predict(essays_TD)
        vd_wd_predictions_by_code = tagger.predict(essays_VD)

        """ Aggregate results """
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)
        pass

    suffix = "_AVG_PERCEPTRON_MOST_COMMON_TAG_HYPER_PARAM_TUNING"
    CB_TAGGING_TD, CB_TAGGING_VD = "CB_TAGGING_TD" + suffix, "CB_TAGGING_VD" + suffix
    parameters = dict(config)
    parameters["prev_tag_sharing"] = True  # don't include tags from other binary models
    """ False: 0.737 - 30 iterations """
    parameters["num_iterations"] = num_iterations
    parameters["tag_history"] = tag_history
    parameters["combo_freq_threshold"] = TAG_FREQ_THRESHOLD

    parameters["extractors"] = extractor_names
    wd_algo = "AveragedPerceptronMultiClass"

    _ = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag,
                                               parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag,
                                               parameters, wd_algo)
    avg_f1 = float(processor.get_metric(CB_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1


""" Prepare the folds (pre-process to avoid unnecessary computation """
# Build this up once
cv_wd_td_ys_by_tag, cv_wd_vd_ys_by_tag = defaultdict(list), defaultdict(list)
# Store the random pickled file names for use in training
k_fold_data_fnames = {}
for kfold, (essays_TD, essays_VD) in enumerate(folds):
    """ Compute the target labels (ys) """
    # Just get the tags (ys)
    _, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    _, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
    wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)

    """ Transform TD to only have most frequent tags """
    essays_TD_most_freq = essaysfeats_to_most_common_tags(essays_TD, tag_freq=tag_freq)

    data_filename = tmp_folder + "fold_%i_avg_prcptron_%s.dill" % (kfold, str(randint(0, 9999999)))
    k_fold_data_fnames[kfold] = data_filename

    k_fold_data = (essays_VD, essays_VD, essays_TD_most_freq, wd_td_ys_bytag, wd_vd_ys_bytag)
    with open(data_filename, "wb+") as f:
        dill.dump(k_fold_data, f)

best_f1 = 0
#for iterations in [10, 20, 30, 40, 50]:
for iterations in [10]:
    #for tag_hist in [0, 1, 3, 5, 10, 15, 20]:
    if iterations == 1:
        tag_history = [0]
    else:
        tag_history = [8, 12, 15]

    for tag_hist in tag_history:
        new_f1 = evaluate_tagger(num_iterations=iterations, tag_history=tag_hist, k_fold_data_fnames=k_fold_data_fnames)
        if new_f1 > best_f1:
            best_f1 = new_f1
            print(("!" * 8) + " NEW BEST MICRO F1 " + ("!" * 8))
        print(" Micro F1 %f for iterations [%i] and tag history [%i]" % (new_f1, iterations, tag_hist))

# Clean up tmp folder
files = find_files(tmp_folder, ".*", False)
for file in files:
    os.remove(file)