import logging

import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from featureextractionfunctions import *
from load_data import load_process_essays, extract_features
from perceptron_tagger_multiclass_combo import PerceptronTaggerMultiClassCombo
from results_procesor import ResultsProcessor
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

NUM_TRAIN_ITERATIONS = 30   # 30 best
TAG_HISTORY          = 10
TAG_FREQ_THRESHOLD   = 5
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
folder =                            root_folder + "Training/"
processed_essay_filename_prefix =   root_folder + "Pickled/essays_proc_pickled_"
features_filename_prefix =          root_folder + "Pickled/feats_pickled_"

out_metrics_file     =              settings.data_directory + "CoralBleaching/Results/metrics.txt"
out_predictions_file =              settings.data_directory + "CoralBleaching/Results/predictions.txt"

config = get_config(folder)

""" FEATURE EXTRACTION """
config["window_size"] = 9
offset = (config["window_size"] - 1) / 2

unigram_bow_window = fact_extract_bow_ngram_features(offset, 1)

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)
trigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 3)

extractors = [unigram_bow_window,
              unigram_window_stemmed,
              biigram_window_stemmed,
              trigram_window_stemmed,
              extract_brown_cluster,
              extract_dependency_relation
]

feat_config = dict(config.items() + [("extractors", extractors)])

""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk
mem_extract_features = memoize_to_disk(filename_prefix=features_filename_prefix)(extract_features)
essay_feats = mem_extract_features(tagged_essays, **feat_config)

logger.info("Features loaded")

""" DEFINE TAGS """
_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
regular_tags = list(set((t for t in flatten(lst_all_tags) if t[0].isdigit())))

wd_train_tags = regular_tags
wd_test_tags  = regular_tags

# tags to evaluate against

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """

# Gather metrics per fold
cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

folds = cross_validation(essay_feats, CV_FOLDS)
def pad_str(val):
    return str(val).ljust(20) + "  "

def toDict(obj):
    return obj.__dict__

#TODO Parallelize
for i,(essays_TD, essays_VD) in enumerate(folds):

    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    print "\nFold %s" % i
    print "Training Tagging Model"
    """ Training """

    """ Compute the target labels (ys) """
    _, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    _, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
    wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)

    # Combo_freq_threshold must be one to do proper LBL Powerset
    tagger = PerceptronTaggerMultiClassCombo(wd_train_tags, tag_history=TAG_HISTORY, combo_freq_threshold=1)
    tagger.train(essays_TD, nr_iter=NUM_TRAIN_ITERATIONS)

    td_wd_predictions_by_code = tagger.predict(essays_TD)
    vd_wd_predictions_by_code = tagger.predict(essays_VD)

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    pass

suffix = "_AVG_PERCEPTRON_LBL_POWERSET"
CB_TAGGING_TD, CB_TAGGING_VD = "CB_TAGGING_TD" + suffix, "CB_TAGGING_VD" + suffix
parameters = dict(config)

parameters["num_iterations"]        = NUM_TRAIN_ITERATIONS
parameters["tag_freq_threshold"]    = TAG_FREQ_THRESHOLD
parameters["tag_history"]           = TAG_HISTORY

#parameters["use_other_class_prev_labels"] = False
parameters["extractors"]        = map(lambda fn: fn.func_name, extractors)

wd_algo = "AveragedPerceptronMulticlassCombo"

wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

print processor.results_to_string(wd_td_objectid,   CB_TAGGING_TD,  wd_vd_objectid,     CB_TAGGING_VD,  "TAGGING")

""" NOTE THIS DOES QUITE A BIT BETTER ON DETECTING THE RESULT CODES, AND A LITTLE BETTER ON THE CAUSE - EFFECT NODES """