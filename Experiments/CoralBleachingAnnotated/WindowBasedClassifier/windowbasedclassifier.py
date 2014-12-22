__author__ = 'simon.hughes'

import numpy as np
from Decorators import timeit, memoize, memoize_to_disk
from BrattEssay import load_bratt_essays
from processessays import process_sentences, process_essays
from wordtagginghelper import flatten_to_wordlevel_feat_tags, get_wordlevel_ys_by_code
from sent_feats_for_stacking import get_sent_feature_for_stacking, CAUSAL_REL, CAUSE_RESULT, RESULT_REL

from featureextractortransformer import FeatureExtractorTransformer
from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from DictionaryHelper import tally_items

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from metric_processing import *
# END Classifiers

import pickle
import Settings
import os

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Settings for essay pre-processing
MIN_SENTENCE_FREQ   = 2        # i.e. df. Note this is calculated BEFORE creating windows
REMOVE_INFREQUENT   = False    # if false, infrequent words are replaced with "INFREQUENT"
SPELLING_CORRECT    = True
STEM                = False    # note this tends to improve matters, but is needed to be on for pos tagging and dep parsing
                               # makes tagging model better but causal model worse
REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
MIN_SENTENCE_LENGTH = 3
REMOVE_STOP_WORDS   = False
REMOVE_PUNCTUATION  = True
LOWER_CASE          = False
# construct unique key using settings for pickling
SPARSE              = True

settings = Settings.Settings()
essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_pickled_"
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
features_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/feats_pickled_"

logger.info("Loading Essays")
mem_load_brat_essays = memoize_to_disk(filename_prefix=essay_filename_prefix)(load_bratt_essays)
essays = mem_load_brat_essays()

logger.info("Processing Essays")
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(process_essays)
tagged_essays = mem_process_essays(min_df=MIN_SENTENCE_FREQ, remove_infrequent=REMOVE_INFREQUENT,
                                       spelling_correct=SPELLING_CORRECT,
                                       replace_nums=REPLACE_NUMS, stem=STEM, remove_stop_words=REMOVE_STOP_WORDS,
                                       remove_punctuation=REMOVE_PUNCTUATION, lower_case=LOWER_CASE)

# most params below exist ONLY for the purposes of the hashing to and from disk
@memoize_to_disk(filename_prefix=features_filename_prefix)
def extract_features(min_df,
                     rem_infreq, spell_correct,
                     replace_nos, stem, rem_stop_wds,
                     rem_punc, l_case,
                     win_size, pos_win_size, min_feat_freq,
                     extractors):
    feature_extractor = FeatureExtractorTransformer(extractors)
    return feature_extractor.transform(tagged_essays)

# FEATURE SETTINGS
WINDOW_SIZE         = 7
POS_WINDOW_SIZE     = 1
MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5
# END FEATURE SETTINGS

offset = (WINDOW_SIZE-1) / 2
unigram_window = fact_extract_positional_word_features(offset)
biigram_window = fact_extract_ngram_features(offset, 2)
pos_tag_window = fact_extract_positional_POS_features((POS_WINDOW_SIZE-1/2))
#TODO - add POS TAGS (positional)
#TODO - add dep parse feats
#extractors = [unigram_window, biigram_window, pos_tag_window]
extractors = [unigram_window, biigram_window]

essay_feats = extract_features(min_df=MIN_SENTENCE_FREQ, rem_infreq=REMOVE_INFREQUENT,
                               spell_correct=SPELLING_CORRECT,
                               replace_nos=REPLACE_NUMS, stem=STEM, rem_stop_wds=REMOVE_STOP_WORDS,
                               rem_punc=REMOVE_PUNCTUATION, l_case=LOWER_CASE,
                               win_size=WINDOW_SIZE, pos_win_size=POS_WINDOW_SIZE,
                               min_feat_freq=MIN_FEAT_FREQ, extractors=extractors)

MIN_TAG_FREQ = 5
_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
""" Get all tags above the frequency above """
flt_lst_tags = flatten(lst_all_tags)
tally_tags = tally_items(flt_lst_tags, freq_threshold=MIN_TAG_FREQ)
all_tags = set(tally_tags.keys())
if "it" in all_tags:
    all_tags.remove("it")

# use more tags for training for sentence level classifier
""" TAGS """
causal_tags = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]

wd_train_tags = list(all_tags)
#wd_test_tags  = [tag for tag in all_tags if tag.isdigit() or tag == "explicit"]
wd_test_tags  = wd_train_tags

# tags from tagging model used to train the stacked model
sent_feat_tags = wd_train_tags
sent_interaction_tags = [tag for tag in all_tags if tag.isdigit() or tag in set(("Causer", "Result", "explicit")) ]

assert "Causer" in sent_feat_tags   , "To extract causal relations, we need Causer tags"
assert "Result" in sent_feat_tags   , "To extract causal relations, we need Result tags"
assert "explicit" in sent_feat_tags , "To extract causal relations, we need explicit tags"
# tags to evaluate against
sent_train_test_tags = wd_train_tags + causal_tags

folds = cross_validation(essay_feats, CV_FOLDS)
"""Word level metrics """
wd_td_wt_mean_prfa, wd_vd_wt_mean_prfa, wd_td_mean_prfa, wd_vd_mean_prfa = [], [], [], []
wd_td_all_metricsByTag, wd_vd_all_metricsByTag = defaultdict(list), defaultdict(list)

"""Sentence level metrics """
sent_td_wt_mean_prfa, sent_vd_wt_mean_prfa, sent_td_mean_prfa, sent_vd_mean_prfa = [], [], [], []
sent_td_all_metricsByTag , sent_vd_all_metricsByTag = defaultdict(list), defaultdict(list)

# Linear SVC seems to do better
#fn_create_cls = lambda: LogisticRegression()
fn_create_wd_cls    = lambda : LinearSVC(C=1.0)
fn_create_sent_cls  = lambda : LinearSVC(C=1.0)

#TODO Parallelize
for i,(TD, VD) in enumerate(folds):

    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    print "\nFold %s" % i
    print "Training Tagging Model"
    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(VD)

    feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE)
    td_X = feature_transformer.fit_transform(td_feats)
    vd_X = feature_transformer.transform(vd_feats)
    td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)

    """ TRAIN Tagger """
    tag2word_classifier = train_classifier_per_code(td_X, td_ys_bytag, fn_create_wd_cls, wd_train_tags)

    """ TEST Tagger """
    td_metricsByTag, td_wt_mean_prfa, td_mean_prfa = test_classifier_per_code(td_X, td_ys_bytag, tag2word_classifier, wd_test_tags)
    vd_metricsByTag, vd_wt_mean_prfa, vd_mean_prfa = test_classifier_per_code(vd_X, vd_ys_bytag, tag2word_classifier, wd_test_tags)

    wd_td_wt_mean_prfa.append(td_wt_mean_prfa), wd_td_mean_prfa.append(td_mean_prfa)
    wd_vd_wt_mean_prfa.append(vd_wt_mean_prfa), wd_vd_mean_prfa.append(vd_mean_prfa)
    merge_metrics(td_metricsByTag, wd_td_all_metricsByTag)
    merge_metrics(vd_metricsByTag, wd_vd_all_metricsByTag)

    print "Training Sentence Model"
    """ SENTENCE LEVEL PREDICTIONS FROM STACKING """
    sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking(sent_feat_tags, sent_interaction_tags, TD, td_X, td_ys_bytag, tag2word_classifier, SPARSE)
    sent_vd_xs, sent_vd_ys_bycode = get_sent_feature_for_stacking(sent_feat_tags, sent_interaction_tags, VD, vd_X, vd_ys_bytag, tag2word_classifier, SPARSE)

    """ Train Stacked Classifier """
    tag2sent_classifier = train_classifier_per_code(sent_td_xs, sent_td_ys_bycode , fn_create_sent_cls, sent_train_test_tags)

    """ Test Stack Classifier """
    s_td_metricsByTag, s_td_wt_mean_prfa, s_td_mean_prfa = test_classifier_per_code(sent_td_xs, sent_td_ys_bycode , tag2sent_classifier, sent_train_test_tags )
    s_vd_metricsByTag, s_vd_wt_mean_prfa, s_vd_mean_prfa = test_classifier_per_code(sent_vd_xs, sent_vd_ys_bycode , tag2sent_classifier, sent_train_test_tags )

    sent_td_wt_mean_prfa.append(s_td_wt_mean_prfa), sent_td_mean_prfa.append(s_td_mean_prfa)
    sent_vd_wt_mean_prfa.append(s_vd_wt_mean_prfa), sent_vd_mean_prfa.append(s_vd_mean_prfa)
    merge_metrics(s_td_metricsByTag, sent_td_all_metricsByTag)
    merge_metrics(s_vd_metricsByTag, sent_vd_all_metricsByTag)

# print results for each code
wd_mean_td_metrics = agg_metrics(wd_td_all_metricsByTag, mean_rpfa)
wd_mean_vd_metrics = agg_metrics(wd_vd_all_metricsByTag, mean_rpfa)

sent_mean_td_metrics = agg_metrics(sent_td_all_metricsByTag, mean_rpfa)
sent_mean_vd_metrics = agg_metrics(sent_vd_all_metricsByTag, mean_rpfa)

print "TAGGING"
print_metrics_for_codes(wd_mean_td_metrics, wd_mean_vd_metrics)

print "\n\nSENTENCE"
print_metrics_for_codes(sent_mean_td_metrics, sent_mean_vd_metrics)

print fn_create_wd_cls()
# print macro measures
print "\nTAGGING"
print "\nTraining   Performance"
print "Weighted:" + str(mean_rpfa(wd_td_wt_mean_prfa))
print "Mean    :" + str(mean_rpfa(wd_td_mean_prfa))

print "\nValidation Performance"
print "Weighted:" + str(mean_rpfa(wd_vd_wt_mean_prfa))
print "Mean    :" + str(mean_rpfa(wd_vd_mean_prfa))

print "\n\n"
print fn_create_wd_cls()
# print macro measures
print "\nSENTENCE"
print "\nTraining   Performance"
print "Weighted:" + str(mean_rpfa(sent_td_wt_mean_prfa))
print "Mean    :" + str(mean_rpfa(sent_td_mean_prfa))

print "\nValidation Performance"
print "Weighted:" + str(mean_rpfa(sent_vd_wt_mean_prfa))
print "Mean    :" + str(mean_rpfa(sent_vd_mean_prfa))

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

#TODO Include dependency parse features
#TODO Parallelize the cross fold validation

VD RESULTS (for params above:
STEMMED
    LinearSVC(C=1.0)
    Weighted:Recall: 0.5933, Precision: 0.6711, F1: 0.6199, Accuracy: 0.9774, Codes:     5
    Mean    :Recall: 0.5532, Precision: 0.6466, F1: 0.5747, Accuracy: 0.9862, Codes:     5

NO STEM
    LinearSVC(C=1.0)
    Weighted:Recall: 0.5843, Precision: 0.6705, F1: 0.6140, Accuracy: 0.9773, Codes:     5
    Mean    :Recall: 0.5425, Precision: 0.6426, F1: 0.5668, Accuracy: 0.9860, Codes:     5

NO STEM + POS (WINDOW 1)
    Weighted:Recall: 0.5848, Precision: 0.6717, F1: 0.6146, Accuracy: 0.9775, Codes:     5
    Mean    :Recall: 0.5447, Precision: 0.6463, F1: 0.5690, Accuracy: 0.9861, Codes:     5

NO STEM + POS (WINDOW 3)
    Weighted:Recall: 0.5848, Precision: 0.6717, F1: 0.6146, Accuracy: 0.9775, Codes:     5
    Mean    :Recall: 0.5447, Precision: 0.6463, F1: 0.5690, Accuracy: 0.9861, Codes:     5


NO STEM + POS TAGS
    LinearSVC(C=1.0)
    Weighted:Recall: 0.5823, Precision: 0.6696, F1: 0.6124, Accuracy: 0.9772, Codes:     5
    Mean    :Recall: 0.5436, Precision: 0.6414, F1: 0.5671, Accuracy: 0.9860, Codes:     5

ONE HOT FEATS DO BETTER!
"""
