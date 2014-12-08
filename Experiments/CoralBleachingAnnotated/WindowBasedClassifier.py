__author__ = 'simon.hughes'

import numpy as np
from Decorators import timeit, memoize, memoize_to_disk
from BrattEssay import load_bratt_essays
from processessays import process_sentences, process_essays
from wordtagginghelper import flatten_to_wordlevel_feat_tags, get_wordlevel_ys_by_code

from featureextractor import FeatureExtractor
from featuretransformer import FeatureTransformer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten

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
STEM                = True
REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
MIN_SENTENCE_LENGTH = 3
REMOVE_STOP_WORDS   = False
REMOVE_PUNCTUATION  = True
LOWER_CASE          = False

# construct unique key using settings for pickling

settings = Settings.Settings()
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/proc_essays_pickled_"
features_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/feats_pickled_"

@memoize_to_disk(filename_prefix=processed_essay_filename_prefix)
def load_and_process_essays(min_df=5,
                           remove_infrequent=False, spelling_correct=True,
                           replace_nums=True, stem=False, remove_stop_words=False,
                           remove_punctuation=True, lower_case=True):

    logger.info("Loading Essays")
    essays = load_bratt_essays()
    logger.info("Processing Essays")
    return process_essays(essays,
                                   min_df=min_df,
                                   remove_infrequent=remove_infrequent,
                                   spelling_correct=spelling_correct,
                                   replace_nums=replace_nums,
                                   stem=stem,
                                   remove_stop_words=remove_stop_words,
                                   remove_punctuation=remove_punctuation,
                                   lower_case=lower_case)

tagged_essays = load_and_process_essays(min_df=MIN_SENTENCE_FREQ, remove_infrequent= REMOVE_INFREQUENT, spelling_correct= SPELLING_CORRECT,
                        replace_nums= REPLACE_NUMS, stem=STEM, remove_stop_words= REMOVE_STOP_WORDS, remove_punctuation= REMOVE_PUNCTUATION, lower_case=LOWER_CASE)

# FEATURE SETTINGS
WINDOW_SIZE         = 7
MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5
# END FEATURE SETTINGS

offset = (WINDOW_SIZE-1) / 2
unigram_window = fact_extract_positional_word_features(offset)
biigram_window = fact_extract_ngram_features(offset, 2)
#TODO - add POS TAGS (positional)
#TODO - add dep parse feats
#TODO - memoize features above for speed
extractors = [unigram_window, biigram_window]

# most params below exist ONLY for the purposes of the hashing to and from disk
@memoize_to_disk(filename_prefix=features_filename_prefix)
def extract_features(min_df,
                     remove_infrequent, spelling_correct,
                     replace_nums, stem, remove_stop_words,
                     remove_punctuation, lower_case,
                     window_size, min_feature_freq,
                     extractors):
    feature_extractor = FeatureExtractor(extractors)
    return feature_extractor.transform(tagged_essays)

essay_feats = extract_features(min_df=MIN_SENTENCE_FREQ, remove_infrequent=REMOVE_INFREQUENT,
                               spelling_correct=SPELLING_CORRECT,
                               replace_nums=REPLACE_NUMS, stem=STEM, remove_stop_words=REMOVE_STOP_WORDS,
                               remove_punctuation=REMOVE_PUNCTUATION, lower_case=LOWER_CASE,

                               window_size=WINDOW_SIZE, min_feature_freq=MIN_FEAT_FREQ, extractors=extractors)

_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
all_tags = set(flatten(lst_all_tags))

# use more tags for training for sentence level classifier
train_tags = [c for c in all_tags if c != "it"]
test_tags  = [c for c in all_tags if c.isdigit() or c == "explicit"]

folds = cross_validation(essay_feats, CV_FOLDS)
lst_td_wt_mean_prfa, lst_vd_wt_mean_prfa, lst_td_mean_prfa, lst_vd_mean_prfa = [], [], [], []
td_all_metricsByTag = defaultdict(list)
vd_all_metricsByTag = defaultdict(list)

def merge_metrics(src, tgt):
    for k, metric in src.items():
        tgt[k].append(metric)

def agg_metrics(src, agg_fn):
    agg = dict()
    for k, metrics in src.items():
        agg[k] = agg_fn(metrics)
    return agg

# Linear SVC seems to do better
#fn_create_cls = lambda: LogisticRegression()
fn_create_cls = lambda : LinearSVC(C=1.0)

for i,(TD, VD) in enumerate(folds):
    print "\nFold %s" % i
    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(VD)

    feature_transformer = FeatureTransformer(min_feature_freq=MIN_FEAT_FREQ)
    td_X = feature_transformer.fit_transform(td_feats)
    vd_X = feature_transformer.transform(vd_feats)
    td_ys_bycode = get_wordlevel_ys_by_code(td_tags)
    vd_ys_bycode = get_wordlevel_ys_by_code(vd_tags)

    """ TRAIN """
    tag2Classifier = train_wordlevel_classifier(td_X, td_ys_bycode, fn_create_cls, train_tags)

    """ TEST """
    td_metricsByTag, td_wt_mean_prfa, td_mean_prfa = test_word_level_classifiers(td_X, td_ys_bycode, tag2Classifier, test_tags)
    vd_metricsByTag, vd_wt_mean_prfa, vd_mean_prfa = test_word_level_classifiers(vd_X, vd_ys_bycode, tag2Classifier, test_tags)

    lst_td_wt_mean_prfa.append(td_wt_mean_prfa), lst_td_mean_prfa.append(td_mean_prfa)
    lst_vd_wt_mean_prfa.append(vd_wt_mean_prfa), lst_vd_mean_prfa.append(vd_mean_prfa)
    merge_metrics(td_metricsByTag, td_all_metricsByTag)
    merge_metrics(vd_metricsByTag, vd_all_metricsByTag)

print fn_create_cls()
# print results for each code
mean_td_metrics = agg_metrics(td_all_metricsByTag, mean_rpfa)
mean_vd_metrics = agg_metrics(vd_all_metricsByTag, mean_rpfa)

print_metrics_for_codes(mean_td_metrics, mean_vd_metrics)
# print macro measures
print "\nTraining   Performance"
print "Weighted:" + str(mean_rpfa(lst_td_wt_mean_prfa))
print "Mean    :" + str(mean_rpfa(lst_td_mean_prfa))

print "\nValidation Performance"
print "Weighted:" + str(mean_rpfa(lst_vd_wt_mean_prfa))
print "Mean    :" + str(mean_rpfa(lst_vd_mean_prfa))

"""
# REWRITE - see FeatureExtractor and FeatureExtractor fns
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

RESULTS (for params above:
"""
