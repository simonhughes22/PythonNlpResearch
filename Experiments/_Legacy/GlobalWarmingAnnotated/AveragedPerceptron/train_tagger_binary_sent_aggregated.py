from Decorators import memoize_to_disk
from GWCodes import GWConceptCodes
from predictions_to_file import predictions_to_file
from sent_feats_for_stacking import *
from load_data import load_process_essays, extract_features

from featureextractionfunctions import *
from CrossValidation import cross_validation
from tag_frequency import get_tag_freq
from wordtagginghelper import *
from IterableFP import flatten
from collections import defaultdict
from tagger_config import get_config
from perceptron_tagger_binary import PerceptronTaggerBinary
from results_procesor import ResultsProcessor
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

NUM_TRAIN_ITERATIONS = 3
TAG_HISTORY          = 0
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "/GlobalWarming/BrattFiles/merged/"
processed_essay_filename_prefix =   folder + "Pickled/essays_proc_pickled_"
features_filename_prefix =          folder + "Pickled/feats_pickled_"

prefix = "Binary_Perceptron_Metrics"
out_metrics_file     =              folder + "Results/" + prefix + "_Metrics.txt"
out_predictions_file =              folder + "Results/" + prefix + "_Predictions.txt"

config = get_config(folder)

""" FEATURE EXTRACTION """
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

#pos_tag_window = fact_extract_positional_POS_features(offset)
#pos_tag_plus_wd_window = fact_extract_positional_POS_features_plus_word(offset)
#head_wd_window = fact_extract_positional_head_word_features(offset)
#pos_dep_vecs = fact_extract_positional_dependency_vectors(offset)

extractors = [unigram_window_stemmed, biigram_window_stemmed]
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
gw_codes = GWConceptCodes()
tag_freq = get_tag_freq(tagged_essays)
freq_tags = list(set((tag for tag, freq in tag_freq.items()
                      if freq >= MIN_TAG_FREQ and gw_codes.is_valid_code(tag))))

non_causal  = [t for t in freq_tags if "->" not in t]
only_causal = [t for t in freq_tags if "->" in t]

_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
regular_tags = list(set((t for t in flatten(lst_all_tags)
                         if "->" in t
                         and gw_codes.is_valid_code(t))))

CAUSE_TAGS = ["Causer", "Result", "explicit"]
CAUSAL_REL_TAGS = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]# + ["explicit"]

"""  """
#
training_tags = list(set(regular_tags + only_causal + CAUSE_TAGS + CAUSAL_REL_TAGS))
wd_test_tags  = training_tags

# tags to evaluate against

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """

f_output_file = open(out_predictions_file, "w+")
f_output_file.write("Essay|Sent Number|Processed Sentence|Concept Codes|Predictions\n")

# Gather metrics per fold
cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

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

    # Just get the tags (ys)
    _, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    _, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)

    wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, training_tags)
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, training_tags)

    tag2word_classifier, td_wd_predictions_by_code, vd_wd_predictions_by_code = {}, {}, {}

    tagger = PerceptronTaggerBinary(training_tags, tag_history=TAG_HISTORY)
    tagger.train(essays_TD, nr_iter=NUM_TRAIN_ITERATIONS)

    td_wd_predictions_by_code = tagger.predict(essays_TD)
    vd_wd_predictions_by_code = tagger.predict(essays_VD)

    """ Store Predictions for computing error metrics """
    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    """ Train Sentence Classification Ensemble Model """

    """ Aggregate word predictions to sentence level """
    td_sent_predictions_by_code = get_sent_tags_from_word_tags(essays_TD, td_wd_predictions_by_code)
    vd_sent_predictions_by_code = get_sent_tags_from_word_tags(essays_VD, vd_wd_predictions_by_code)

    """ get sentence level labels """
    sent_td_ys_bycode = get_sent_tags_from_word_tags(essays_TD, wd_td_ys_bytag)
    sent_vd_ys_bycode = get_sent_tags_from_word_tags(essays_VD, wd_vd_ys_bytag)

    merge_dictionaries(sent_td_ys_bycode, cv_sent_td_ys_by_tag)
    merge_dictionaries(sent_vd_ys_bycode, cv_sent_vd_ys_by_tag)
    merge_dictionaries(td_sent_predictions_by_code, cv_sent_td_predictions_by_tag)
    merge_dictionaries(vd_sent_predictions_by_code, cv_sent_vd_predictions_by_tag)

    predictions_to_file(f_output_file, sent_vd_ys_bycode, vd_sent_predictions_by_code, essays_VD, codes=training_tags)

    print("STOPPING ONE ONE FOLD FOR TESTING")
    break
pass

CB_TAGGING_TD, CB_TAGGING_VD = "GW_TAGGING_TD", "GW_TAGGING_VD"
parameters = dict(config)
parameters["prev_tag_sharing"] = True # don't include tags from other binary models
""" False: 0.737 - 30 iterations """
parameters["num_iterations"] = NUM_TRAIN_ITERATIONS
parameters["right-to-left"] = True
parameters["tag_history"]    = TAG_HISTORY
#parameters["AverageWeights"] = False # Bad - averaging really helps
parameters["extractors"] = map(lambda fn: fn.func_name, extractors)

wd_algo = "AveragedPerceptronBinary"

wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

print processor.results_to_string(wd_td_objectid, CB_TAGGING_TD, wd_vd_objectid, CB_TAGGING_VD,  "TAGGING")
