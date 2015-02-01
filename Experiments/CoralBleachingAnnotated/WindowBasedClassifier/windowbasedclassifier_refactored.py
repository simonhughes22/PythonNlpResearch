__author__ = 'simon.hughes'

import numpy as np
from Decorators import memoize_to_disk
from BrattEssay import load_bratt_essays
from processessays import process_sentences, process_essays
from sent_feats_for_stacking import *
from load_data import load_process_essays, extract_features

from featureextractortransformer import FeatureExtractorTransformer
from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from DictionaryHelper import tally_items
from metric_processing import *
from predictions_to_file import predictions_to_file
from result_processing import get_results

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.lda import LDA

# END Classifiers

import pickle
import Settings
import os

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Settings for loading essays
INCLUDE_VAGUE       = True
INCLUDE_NORMAL      = False

# Settings for essay pre-processing
MIN_SENTENCE_FREQ   = 2        # i.e. df. Note this is calculated BEFORE creating windows
REMOVE_INFREQUENT   = False    # if false, infrequent words are replaced with "INFREQUENT"
SPELLING_CORRECT    = True
STEM                = False     # note this tends to improve matters, but is needed to be on for pos tagging and dep parsing
                               # makes tagging model better but causal model worse
REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
MIN_SENTENCE_LENGTH = 3
REMOVE_STOP_WORDS   = False
REMOVE_PUNCTUATION  = True
LOWER_CASE          = False
# construct unique key using settings for pickling

settings = Settings.Settings()
folder = settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
#folder = settings.data_directory + "CoralBleaching/BrattData/Merged/"
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
features_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/feats_pickled_"

out_metrics_file     = settings.data_directory + "CoralBleaching/Results/metrics.txt"
out_predictions_file = settings.data_directory + "CoralBleaching/Results/predictions.txt"

mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays(    folder=folder, min_df=MIN_SENTENCE_FREQ, remove_infrequent=REMOVE_INFREQUENT,
                                       spelling_correct=SPELLING_CORRECT,
                                       replace_nums=REPLACE_NUMS, stem=STEM, remove_stop_words=REMOVE_STOP_WORDS,
                                       remove_punctuation=REMOVE_PUNCTUATION, lower_case=LOWER_CASE,
                                       include_vague=INCLUDE_VAGUE, include_normal=INCLUDE_NORMAL)

# FEATURE SETTINGS
WINDOW_SIZE         = 7
POS_WINDOW_SIZE     = 1
MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5
# END FEATURE SETTINGS

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True
MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

offset = (WINDOW_SIZE-1) / 2
unigram_window = fact_extract_positional_word_features(offset)
biigram_window = fact_extract_ngram_features(offset, 2)

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

pos_tag_window = fact_extract_positional_POS_features(offset)
pos_tag_plus_wd_window = fact_extract_positional_POS_features_plus_word(offset)
head_wd_window = fact_extract_positional_head_word_features(offset)

extractors = [unigram_window_stemmed, biigram_window_stemmed]

if pos_tag_window in extractors and STEM:
    raise Exception("POS tagging won't work with stemming on")

# most params below exist ONLY for the purposes of the hashing to and from disk
mem_extract_features = memoize_to_disk(filename_prefix=features_filename_prefix)(extract_features)
essay_feats = mem_extract_features(tagged_essays,
                               folder=folder, extractors=extractors,
                               min_df=MIN_SENTENCE_FREQ, rem_infreq=REMOVE_INFREQUENT,
                               sp_crrct=SPELLING_CORRECT,
                               replace_nos=REPLACE_NUMS, stem=STEM, rem_stop_wds=REMOVE_STOP_WORDS,
                               rem_punc=REMOVE_PUNCTUATION, lcase=LOWER_CASE,
                               win_size=WINDOW_SIZE, pos_win_size=POS_WINDOW_SIZE,
                               min_ft_freq=MIN_FEAT_FREQ,
                               inc_vague=INCLUDE_VAGUE, inc_normal=INCLUDE_NORMAL)

_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
""" Get all tags above the frequency above """
""" NOTE WHEN OUTPUTING RESULTS WE NEED TO USE ALL TAGS, NOT HIGHER FREQ TAGS """
flt_lst_tags = flatten(lst_all_tags)
tally_tags = tally_items(flt_lst_tags, freq_threshold=MIN_TAG_FREQ)
all_tags_above_threshold = set(tally_tags.keys())
if "it" in all_tags_above_threshold:
    all_tags_above_threshold.remove("it")

# use more tags for training for sentence level classifier
""" !!! THE TAGS USED TO RUN !!! """
regular_tags = [t for t in all_tags_above_threshold if t[0].isdigit()]
cause_tags = ["Causer", "Result", "explicit"]
causal_rel_tags = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]# + ["explicit"]

#wd_train_tags = list(all_tags_above_threshold)
""" works best with all the pair-wise causal relation codes """
#wd_train_tags = list(all_tags_above_threshold.union(cause_tags))
#wd_train_tags = regular_tags
wd_train_tags = regular_tags + cause_tags
#wd_test_tags  = [tag for tag in all_tags if tag.isdigit() or tag == "explicit"]
wd_test_tags  = regular_tags + cause_tags

# tags from tagging model used to train the stacked model
sent_input_feat_tags = wd_train_tags
# find interactions between these predicted tags from the word tagger to feed to the sentence tagger
#sent_input_interaction_tags = [tag for tag in all_tags_above_threshold if tag.isdigit() or tag in set(("Causer", "Result", "explicit")) ]
sent_input_interaction_tags = regular_tags + cause_tags
# tags to train (as output) for the sentence based classifier
sent_output_train_test_tags = list(set(regular_tags + causal_rel_tags))
#sent_output_train_test_tags = list(set(regular_tags))

assert "Causer" in sent_input_feat_tags   , "To extract causal relations, we need Causer tags"
assert "Result" in sent_input_feat_tags   , "To extract causal relations, we need Result tags"
assert "explicit" in sent_input_feat_tags , "To extract causal relations, we need explicit tags"
# tags to evaluate against

folds = cross_validation(essay_feats, CV_FOLDS)
"""Word level metrics """
wd_td_wt_mean_prfa, wd_vd_wt_mean_prfa, wd_td_mean_prfa, wd_vd_mean_prfa = [], [], [], []
wd_td_all_metricsByTag, wd_vd_all_metricsByTag = defaultdict(list), defaultdict(list)

"""Sentence level metrics """
sent_td_wt_mean_prfa, sent_vd_wt_mean_prfa, sent_td_mean_prfa, sent_vd_mean_prfa = [], [], [], []
sent_td_all_metricsByTag , sent_vd_all_metricsByTag = defaultdict(list), defaultdict(list)

""" Log Reg + Log Reg is best!!! """
# NOTE - GBT is stochastic in the SPLITS, and so you will get non-deterministic results
fn_create_wd_cls = lambda: LogisticRegression() # C=1, dual = False seems optimal
#fn_create_wd_cls    = lambda : LinearSVC(C=1.0)
#fn_create_sent_cls  = lambda : LinearSVC(C=1.0)
fn_create_sent_cls  = lambda : LogisticRegression(dual=True) # C around 1.0 seems pretty optimal
#fn_create_sent_cls  = lambda : GradientBoostingClassifier() #F1 = 0.5312 on numeric + 5b + casual codes for sentences

if type(fn_create_sent_cls()) == GradientBoostingClassifier:
    SPARSE_SENT_FEATS = False

f_output_file = open(out_predictions_file, "w+")
f_output_file.write("Essay|Sent Number|Processed Sentence|Concept Codes|Predictions\n")
#TODO Parallelize
for i,(essays_TD, essays_VD) in enumerate(folds):

    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    print "\nFold %s" % i
    print "Training Tagging Model"
    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)

    feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
    td_X = feature_transformer.fit_transform(td_feats)
    vd_X = feature_transformer.transform(vd_feats)
    td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)

    """ TRAIN Tagger """
    tag2word_classifier = train_classifier_per_code(td_X, td_ys_bytag, fn_create_wd_cls, wd_train_tags)

    """ TEST Tagger """
    td_metricsByTag, td_wt_mean_prfa, td_mean_prfa, td_wd_predictions_by_code = test_classifier_per_code(td_X, td_ys_bytag, tag2word_classifier, wd_test_tags)
    vd_metricsByTag, vd_wt_mean_prfa, vd_mean_prfa, vd_wd_predictions_by_code = test_classifier_per_code(vd_X, vd_ys_bytag, tag2word_classifier, wd_test_tags)

    wd_td_wt_mean_prfa.append(td_wt_mean_prfa), wd_td_mean_prfa.append(td_mean_prfa)
    wd_vd_wt_mean_prfa.append(vd_wt_mean_prfa), wd_vd_mean_prfa.append(vd_mean_prfa)
    merge_metrics(td_metricsByTag, wd_td_all_metricsByTag)
    merge_metrics(vd_metricsByTag, wd_vd_all_metricsByTag)

    print "Training Sentence Model"
    """ SENTENCE LEVEL PREDICTIONS FROM STACKING """
    sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking(sent_input_feat_tags, sent_input_interaction_tags, essays_TD, td_X, td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)
    sent_vd_xs, sent_vd_ys_bycode = get_sent_feature_for_stacking(sent_input_feat_tags, sent_input_interaction_tags, essays_VD, vd_X, vd_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

    """ Train Stacked Classifier """
    tag2sent_classifier = train_classifier_per_code(sent_td_xs, sent_td_ys_bycode , fn_create_sent_cls, sent_output_train_test_tags)

    """ Test Stack Classifier """
    s_td_metricsByTag, s_td_wt_mean_prfa, s_td_mean_prfa, td_sent_predictions_by_code \
        = test_classifier_per_code(sent_td_xs, sent_td_ys_bycode , tag2sent_classifier, sent_output_train_test_tags )

    s_vd_metricsByTag, s_vd_wt_mean_prfa, s_vd_mean_prfa, vd_sent_predictions_by_code \
        = test_classifier_per_code(sent_vd_xs, sent_vd_ys_bycode , tag2sent_classifier, sent_output_train_test_tags )

    sent_td_wt_mean_prfa.append(s_td_wt_mean_prfa), sent_td_mean_prfa.append(s_td_mean_prfa)
    sent_vd_wt_mean_prfa.append(s_vd_wt_mean_prfa), sent_vd_mean_prfa.append(s_vd_mean_prfa)
    merge_metrics(s_td_metricsByTag, sent_td_all_metricsByTag)
    merge_metrics(s_vd_metricsByTag, sent_vd_all_metricsByTag)

    predictions_to_file(f_output_file, sent_vd_ys_bycode, vd_sent_predictions_by_code, essays_VD, list(all_tags_above_threshold) + causal_rel_tags)

f_output_file.close()
# print results for each code
s_results = get_results(wd_td_all_metricsByTag, wd_vd_all_metricsByTag, sent_td_all_metricsByTag,
                        sent_vd_all_metricsByTag,
                        wd_td_wt_mean_prfa, wd_td_mean_prfa, wd_vd_wt_mean_prfa, wd_vd_mean_prfa,
                        sent_td_wt_mean_prfa, sent_td_mean_prfa, sent_vd_wt_mean_prfa, sent_vd_mean_prfa,
                        fn_create_wd_cls, fn_create_sent_cls)

print s_results
with open(out_metrics_file, "w+") as f:
    f.write(s_results)
"""
# PLAN
#   WORD LEVEL FEATURE EXTRACTION - use functions specific to the individual word, but that can look around at the
#       previous and next words and sentences if needed. This can handle every scenario where I want to leverage features
#       across sentences and at the essay level.
#   MEMOIZE SENTENCE LEVEL FEATS (e.g. deps) -  Will need memoizing when extracting dependency parse features per sentence (as called once for every word in sentence)
#   WORD \ SENTENCE PARTITIONING FOR WORD AND SENTENCE LEVEL TAGGING
#       Need a class that can transform the feature dictionaries (from essay structure form) into training and test data
#       for word tagging and also for sentence classifying. Suggest do k fold cross validation at the essay level.
#   LOAD RESULTS INTO A DB

#TODO Feed into the sentence classifier the number of words tagged with each category, the proportion of words (to control for sentence length variations) and
    also the number of contiguous segments of each in case some codes occur more than once (in separate segments - probably with > 1 word gaps in between)
#TODO Switch to micro and macro-average F1 scores as described in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf, page 6.
#TODO Look at using the StructureLINK relations (related to anaphora and other devices, see ReadMe.txt in the CoralBleaching folder).
#TODO Include dependency parse features
#TODO Parallelize the cross fold validation


>>> These results compute the mean across all regular tags (tagging model) and regular tags plus causal (sentence)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)


TAGGING

Training   Performance
Weighted:Recall: 0.8021, Precision: 0.9443, F1: 0.8617, Accuracy: 0.9933, Codes:  4175
Mean    :Recall: 0.6790, Precision: 0.9194, F1: 0.7594, Accuracy: 0.9965, Codes:    13

Validation Performance
Weighted:Recall: 0.6375, Precision: 0.8561, F1: 0.7109, Accuracy: 0.9862, Codes:  5219
Mean    :Recall: 0.5141, Precision: 0.7584, F1: 0.5745, Accuracy: 0.9931, Codes:    65


GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)

SENTENCE

Training   Performance
Weighted:Recall: 0.9924, Precision: 0.9875, F1: 0.9899, Accuracy: 0.9966, Codes:  2136
Mean    :Recall: 0.9947, Precision: 0.9858, F1: 0.9900, Accuracy: 0.9984, Codes:    16

Validation Performance
Weighted:Recall: 0.7381, Precision: 0.8234, F1: 0.7679, Accuracy: 0.9359, Codes:  2670
Mean    :Recall: 0.6790, Precision: 0.7513, F1: 0.6872, Accuracy: 0.9662, Codes:    80

****************************************************************************************

LR as the tagger (same tagging results)

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)


SENTENCE

Training   Performance
Weighted:Recall: 0.9575, Precision: 0.9520, F1: 0.9533, Accuracy: 0.9823, Codes:  2136
Mean    :Recall: 0.9788, Precision: 0.9660, F1: 0.9715, Accuracy: 0.9924, Codes:    16

Validation Performance
Weighted:Recall: 0.7399, Precision: 0.8122, F1: 0.7662, Accuracy: 0.9327, Codes:  2670
Mean    :Recall: 0.6717, Precision: 0.7553, F1: 0.6885, Accuracy: 0.9650, Codes:    80


****************************************************************************************
BEST !!!!!

LR as the tagger (same tagging results)
LR as the sentence classifier

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)


SENTENCE

Training   Performance
Weighted:Recall: 0.9560, Precision: 0.9586, F1: 0.9572, Accuracy: 0.9845, Codes:  2136
Mean    :Recall: 0.9735, Precision: 0.9686, F1: 0.9708, Accuracy: 0.9932, Codes:    16

Validation Performance
Weighted:Recall: 0.7383, Precision: 0.8319, F1: 0.7749, Accuracy: 0.9365, Codes:  2670
Mean    :Recall: 0.6726, Precision: 0.7843, F1: 0.7033, Accuracy: 0.9673, Codes:    80

LR + LR
# DEPENDENCIES:

# head only
Validation Performance
Weighted:Recall: 0.7400, Precision: 0.8369, F1: 0.7776, Accuracy: 0.9370, Codes:  2670
Mean    :Recall: 0.6743, Precision: 0.7897, F1: 0.7064, Accuracy: 0.9677, Codes:    80

# head word only
***Validation Performance
Weighted:Recall: 0.7433, Precision: 0.8356, F1: 0.7787, Accuracy: 0.9383, Codes:  2670
Mean    :Recall: 0.6688, Precision: 0.7854, F1: 0.6995, Accuracy: 0.9679, Codes:    80

# head + target
Validation Performance
Weighted:Recall: 0.7382, Precision: 0.8313, F1: 0.7747, Accuracy: 0.9364, Codes:  2670
Mean    :Recall: 0.6667, Precision: 0.7863, F1: 0.7015, Accuracy: 0.9672, Codes:    80

# head only + child only
Validation Performance
Weighted:Recall: 0.7324, Precision: 0.8304, F1: 0.7703, Accuracy: 0.9358, Codes:  2670
Mean    :Recall: 0.6632, Precision: 0.7859, F1: 0.6962, Accuracy: 0.9668, Codes:    80

# child only
Validation Performance
Weighted:Recall: 0.7325, Precision: 0.8306, F1: 0.7714, Accuracy: 0.9359, Codes:  2670
Mean    :Recall: 0.6658, Precision: 0.7843, F1: 0.6983, Accuracy: 0.9669, Codes:    80

#child words only
Validation Performance
Weighted:Recall: 0.7316, Precision: 0.8343, F1: 0.7713, Accuracy: 0.9364, Codes:  2670
Mean    :Recall: 0.6624, Precision: 0.7859, F1: 0.6959, Accuracy: 0.9670, Codes:    80


# children + target
Validation Performance
Weighted:Recall: 0.7383, Precision: 0.8307, F1: 0.7744, Accuracy: 0.9362, Codes:  2670
Mean    :Recall: 0.6680, Precision: 0.7847, F1: 0.7012, Accuracy: 0.9672, Codes:    80

# head -> child
Validation Performance
Weighted:Recall: 0.7381, Precision: 0.8308, F1: 0.7745, Accuracy: 0.9356, Codes:  2670
Mean    :Recall: 0.6736, Precision: 0.7841, F1: 0.7031, Accuracy: 0.9670, Codes:    80

!!! FORGOT TO TURN OFF STEMMING !!!

@@@ 0.7591 @@@@

Regular NO STEM
Validation Performance
Weighted:Recall: 0.7198, Precision: 0.8226, F1: 0.7591, Accuracy: 0.9325, Codes:  2670
Mean    :Recall: 0.6534, Precision: 0.7761, F1: 0.6860, Accuracy: 0.9652, Codes:    80

#POSITIONAL HEAD
Validation Performance
Weighted:Recall: 0.7071, Precision: 0.8208, F1: 0.7517, Accuracy: 0.9309, Codes:  2670
Mean    :Recall: 0.6204, Precision: 0.7653, F1: 0.6599, Accuracy: 0.9643, Codes:    80

# DEP HEAD WORD ONLY
Validation Performance
Weighted:Recall: 0.7184, Precision: 0.8205, F1: 0.7579, Accuracy: 0.9324, Codes:  2670
Mean    :Recall: 0.6496, Precision: 0.7714, F1: 0.6825, Accuracy: 0.9649, Codes:    80

# NLTK POS TAGS
Validation Performance
Weighted:Recall: 0.7145, Precision: 0.8198, F1: 0.7538, Accuracy: 0.9316, Codes:  2670
Mean    :Recall: 0.6430, Precision: 0.7808, F1: 0.6789, Accuracy: 0.9647, Codes:    80

#NLT TAG ONLY (unary tag)
**** Validation Performance
Weighted:Recall: 0.7196, Precision: 0.8252, F1: 0.7596, Accuracy: 0.9330, Codes:  2670
Mean    :Recall: 0.6452, Precision: 0.7770, F1: 0.6808, Accuracy: 0.9654, Codes:    80

#NLTK POS + WORD
*****Validation Performance
Weighted:Recall: 0.7262, Precision: 0.8244, F1: 0.7636, Accuracy: 0.9338, Codes:  2670
Mean    :Recall: 0.6611, Precision: 0.7780, F1: 0.6913, Accuracy: 0.9658, Codes:    80

SWITCH TO STEMMING DURING FEAT EXTRACTION FOR THE MAIN INPUTS
    - Leaving sentence untouched for dep parser

STEMMED -

LR + LR + main feats only
Training   Performance
Weighted:Recall: 0.9563, Precision: 0.9597, F1: 0.9579, Accuracy: 0.9846, Codes:  2136
Mean    :Recall: 0.9745, Precision: 0.9704, F1: 0.9721, Accuracy: 0.9933, Codes:    16

Validation Performance
Weighted:Recall: 0.7378, Precision: 0.8306, F1: 0.7742, Accuracy: 0.9361, Codes:  2670
Mean    :Recall: 0.6742, Precision: 0.7848, F1: 0.7058, Accuracy: 0.9672, Codes:    80

add dual regularization to second LR
Validation Performance
Weighted:Recall: 0.7378, Precision: 0.8311, F1: 0.7744, Accuracy: 0.9362, Codes:  2670
Mean    :Recall: 0.6742, Precision: 0.7851, F1: 0.7059, Accuracy: 0.9672, Codes:    80


#NLTK POS + WD (worse with stemming on than stemming alone)
Validation Performance
Weighted:Recall: 0.7389, Precision: 0.8286, F1: 0.7737, Accuracy: 0.9353, Codes:  2670
Mean    :Recall: 0.6770, Precision: 0.7804, F1: 0.7063, Accuracy: 0.9670, Codes:    80

#NLTK Positional POS + WD
Validation Performance
Weighted:Recall: 0.7339, Precision: 0.8291, F1: 0.7696, Accuracy: 0.9345, Codes:  2670
Mean    :Recall: 0.6716, Precision: 0.7869, F1: 0.7021, Accuracy: 0.9665, Codes:    80

#Switch on dual Regularization on sentence classifier - slight performance boost

#Brown Cluster Only
Training   Performance
Weighted:Recall: 0.9593, Precision: 0.9621, F1: 0.9606, Accuracy: 0.9856, Codes:  2136
Mean    :Recall: 0.9766, Precision: 0.9712, F1: 0.9737, Accuracy: 0.9937, Codes:    16

Validation Performance
Weighted:Recall: 0.7365, Precision: 0.8238, F1: 0.7703, Accuracy: 0.9344, Codes:  2670
Mean    :Recall: 0.6712, Precision: 0.7806, F1: 0.7008, Accuracy: 0.9665, Codes:    80

#Brow Cluster + word
Validation Performance
Weighted:Recall: 0.7394, Precision: 0.8287, F1: 0.7739, Accuracy: 0.9355, Codes:  2670
Mean    :Recall: 0.6770, Precision: 0.7804, F1: 0.7063, Accuracy: 0.9670, Codes:    80

"""
