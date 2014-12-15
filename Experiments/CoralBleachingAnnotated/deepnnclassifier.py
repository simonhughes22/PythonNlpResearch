__author__ = 'simon.hughes'

import numpy as np
from Decorators import timeit, memoize, memoize_to_disk
from BrattEssay import load_bratt_essays
from processessays import process_essays

from wordprojectortransformer import WordProjectorTransformer
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
import graphlab

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
REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
MIN_SENTENCE_LENGTH = 3
REMOVE_STOP_WORDS   = False
REMOVE_PUNCTUATION  = True
LOWER_CASE          = True

# construct unique key using settings for pickling

settings = Settings.Settings()
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/proc_essays_pickled_"

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
CV_FOLDS            = 5
# END FEATURE SETTINGS
offset = (WINDOW_SIZE-1) / 2

# don't memoize as it's massive and also fast
word_projector_transformer = WordProjectorTransformer(offset)
essay_feats = word_projector_transformer.transform(tagged_essays)

_, lst_all_tags = flatten_to_wordlevel_vectors_tags(essay_feats)
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
#TODO DNN
fn_create_cls = lambda : LinearSVC(C=1.0)

for i,(TD, VD) in enumerate(folds):
    print "\nFold %s" % i
    """ Data Partitioning and Training """
    td_X, td_tags = flatten_to_wordlevel_vectors_tags(TD)
    vd_X, vd_tags = flatten_to_wordlevel_vectors_tags(VD)

    td_ys_bycode = get_wordlevel_ys_by_code(td_tags)
    vd_ys_bycode = get_wordlevel_ys_by_code(vd_tags)

    """ TRAIN """
    #TODO DNN
    tag2Classifier = train_wordlevel_classifier(td_X, td_ys_bycode, fn_create_cls, train_tags)

    """ TEST """
    td_metricsByTag, td_wt_mean_prfa, td_mean_prfa = test_word_level_classifiers(td_X, td_ys_bycode, tag2Classifier, test_tags)
    vd_metricsByTag, vd_wt_mean_prfa, vd_mean_prfa = test_word_level_classifiers(vd_X, vd_ys_bycode, tag2Classifier, test_tags)

    lst_td_wt_mean_prfa.append(td_wt_mean_prfa), lst_td_mean_prfa.append(td_mean_prfa)
    lst_vd_wt_mean_prfa.append(vd_wt_mean_prfa), lst_vd_mean_prfa.append(vd_mean_prfa)
    merge_metrics(td_metricsByTag, td_all_metricsByTag)
    merge_metrics(vd_metricsByTag, vd_all_metricsByTag)

# print results for each code
mean_td_metrics = agg_metrics(td_all_metricsByTag, mean_rpfa)
mean_vd_metrics = agg_metrics(vd_all_metricsByTag, mean_rpfa)

print_metrics_for_codes(mean_td_metrics, mean_vd_metrics)
print fn_create_cls()
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
#   USE SPARSITY IF GL SUPPORTS IT
#   LOAD RESULTS INTO A DB

#TODO Include dependency parse features

VD RESULTS (for params above:

"""
