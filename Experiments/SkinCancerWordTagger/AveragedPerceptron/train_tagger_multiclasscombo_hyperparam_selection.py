# coding=utf-8
import logging
from collections import Counter

from joblib import Parallel
from joblib import delayed

import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from featureextractionfunctions import *
from load_data import load_process_essays, extract_features
from perceptron_tagger_multiclass_combo import PerceptronTaggerMultiClassCombo
from results_procesor import ResultsProcessor, __MICRO_F1__
from window_based_tagger_config import get_config
from wordtagginghelper import *

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
#TAG_FREQ_THRESHOLD   = 5

# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()
root_folder = settings.data_directory + "SkinCancer/Thesis_Dataset/"
folder =                            root_folder + "Training/"
processed_essay_filename_prefix =   root_folder + "Pickled/essays_proc_pickled_"
features_filename_prefix =          root_folder + "Pickled/feats_pickled_"

config = get_config(folder)

""" FEATURE EXTRACTION """
config["window_size"] = 9
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed  = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed  = fact_extract_ngram_features_stemmed(offset, 2)
triigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 3)
unigram_bow_window      = fact_extract_bow_ngram_features(offset, 1)

#optimal SC feature set
extractors = [unigram_bow_window,
              unigram_window_stemmed,
              biigram_window_stemmed,
              #trigram_window_stemmed,
              extract_brown_cluster,
              #extract_dependency_relation
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

""" Prepare the folds (pre-process to avoid unnecessary computation) """
# Build this up once
cv_wd_td_ys_by_tag, cv_wd_vd_ys_by_tag = defaultdict(list), defaultdict(list)
# Store the random pickled file names for use in training
k_fold_2data = {}
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

    k_fold_2data[kfold] = (essays_TD, essays_VD, essays_TD_most_freq, wd_td_ys_bytag, wd_vd_ys_bytag)

def evaluate_tagger_on_fold(kfold, wd_train_tags, use_tag_features, num_iterations, tag_history):

    logger.info("Loading data for fold %i" % kfold)
    k_fold_data = k_fold_2data[kfold]
    essays_TD, essays_VD, essays_TD_most_freq, wd_td_ys_bytag, wd_vd_ys_bytag = k_fold_data

    """ TRAINING """
    tagger = PerceptronTaggerMultiClassCombo(wd_train_tags, tag_history=tag_history,
                                             combo_freq_threshold=1, use_tag_features=use_tag_features)
    tagger.train(essays_TD_most_freq, nr_iter=num_iterations, verbose=False)

    """ PREDICT """
    td_wd_predictions_by_code = tagger.predict(essays_TD)
    vd_wd_predictions_by_code = tagger.predict(essays_VD)

    logger.info("Fold %i finished" % kfold)
    """ Aggregate results """
    return kfold, td_wd_predictions_by_code, vd_wd_predictions_by_code

def evaluate_tagger(wd_train_tags, use_tag_features, num_iterations, tag_history):

    """ Run K Fold CV in parallel """
    print("\nNew Run - Use Tag Feats: '%s'\t Num Iterations: %i \t Tag History: %i" \
          % (str(use_tag_features), num_iterations, tag_history))

    results = Parallel(n_jobs=(CV_FOLDS))(
         delayed(evaluate_tagger_on_fold)(kfold, wd_train_tags, use_tag_features, num_iterations, tag_history)
            for kfold in range(CV_FOLDS))

    # Merge results of parallel processing
    cv_wd_td_predictions_by_tag, cv_wd_vd_predictions_by_tag    = defaultdict(list), defaultdict(list)
    # important to sort by k value
    for kfold, td_wd_predictions_by_code, vd_wd_predictions_by_code in sorted(results, key = lambda (k, td, vd): k):
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)
        pass

    suffix = "_AVG_PERCEPTRON_MOST_COMMON_TAG_HYPER_PARAM_TUNING"
    SC_TAGGING_TD, SC_TAGGING_VD = "SC_TAGGING_TD" + suffix, "SC_TAGGING_VD" + suffix
    parameters = dict(config)
    parameters["prev_tag_sharing"] = True  # don't include tags from other binary models
    """ False: 0.737 - 30 iterations """
    parameters["use_tag_features"] = str(use_tag_features).lower()
    parameters["num_iterations"] = num_iterations
    parameters["tag_history"] = tag_history
    #parameters["combo_freq_threshold"] = TAG_FREQ_THRESHOLD

    parameters["extractors"] = extractor_names
    wd_algo = "AveragedPerceptronMultiClass_TagHistoryFixed"

    _              = processor.persist_results(SC_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(SC_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

    avg_f1 = float(processor.get_metric(SC_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1

best_f1 = 0
for num_iterations in [1, 2, 3]:          # Number of training iterations before stopping - Should we use early stopping instead?
#for num_iterations in [1, 2, 3, 5, 10, 20, 40]:          # Number of training iterations before stopping - Should we use early stopping instead?

    if num_iterations == 1:
        # For just one iteration, not point in computing all of the history based measure, as predictions are too noisy
        p_tag_history = [0]
        p_use_tag_features  = [False]
    else:
        p_tag_history = [0, 1, 2, 3, 5, 10, 20]
        p_use_tag_features = [False, True]

    for use_tag_feat in p_use_tag_features:         # Whether or not to use the various features used to look at combinations of words with prior tags
        for tag_hist in p_tag_history:              # Tag history used to train the classifier

            new_f1 = evaluate_tagger(wd_train_tags=wd_train_tags, use_tag_features=use_tag_feat, num_iterations=num_iterations, tag_history=tag_hist)
            if new_f1 > best_f1:
                best_f1 = new_f1
                print(("!" * 8) + " NEW BEST MICRO F1 " + ("!" * 8))
            print(" Micro F1 %f - Use Tag Feats: '%s'\t Num Iterations: %i \t Tag History: %i" % (str(use_tag_feat), num_iterations, tag_hist))
