__author__ = 'simon.hughes'

import numpy as np
from Decorators import timeit, memoize, memoize_to_disk
from BrattEssay import load_bratt_essays
from processessays import process_sentences, process_essays
from wordtagginghelper import flatten_to_wordlevel_feat_tags, get_wordlevel_ys_by_code
from sent_feats_for_stacking import *

from featureextractortransformer import FeatureExtractorTransformer
from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from DictionaryHelper import tally_items
from Rpfa import mean_rpfa, weighted_mean_rpfa
from metric_processing import *
from predictions_to_file import predictions_to_file
from result_processing import get_results

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

# Settings for loading essays
INCLUDE_VAGUE       = True
INCLUDE_NORMAL      = False

# Settings for essay pre-processing
MIN_SENTENCE_FREQ   = 2        # i.e. df. Note this is calculated BEFORE creating windows
REMOVE_INFREQUENT   = False    # if false, infrequent words are replaced with "INFREQUENT"
SPELLING_CORRECT    = True
STEM                = True     # note this tends to improve matters, but is needed to be on for pos tagging and dep parsing
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
essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_pickled_"
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
features_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/feats_pickled_"

out_metrics_file     = settings.data_directory + "CoralBleaching/Results/metrics.txt"
out_predictions_file = settings.data_directory + "CoralBleaching/Results/predictions.txt"

logger.info("Loading Essays")
@memoize_to_disk(filename_prefix=essay_filename_prefix)
def load_essays(include_vague=INCLUDE_VAGUE, include_normal=INCLUDE_NORMAL):
    return load_bratt_essays(directory=folder, include_vague=include_vague, include_normal=include_normal)

essays = load_essays(include_vague=INCLUDE_VAGUE, include_normal=INCLUDE_NORMAL)

logger.info("Processing Essays")
@memoize_to_disk(filename_prefix=processed_essay_filename_prefix)
def mem_process_essays(min_df=MIN_SENTENCE_FREQ, remove_infrequent=REMOVE_INFREQUENT,
                    spelling_correct=SPELLING_CORRECT,
                    replace_nums=REPLACE_NUMS, stem=STEM, remove_stop_words=REMOVE_STOP_WORDS,
                    remove_punctuation=REMOVE_PUNCTUATION, lower_case=LOWER_CASE,
                    include_vague=INCLUDE_VAGUE, include_normal=INCLUDE_NORMAL):
    return process_essays(essays, min_df=min_df, remove_infrequent=remove_infrequent, spelling_correct=spelling_correct,
                replace_nums=replace_nums, stem=stem, remove_stop_words=remove_stop_words,
                remove_punctuation=remove_punctuation,lower_case=lower_case)

tagged_essays = mem_process_essays(min_df=MIN_SENTENCE_FREQ, remove_infrequent=REMOVE_INFREQUENT,
                                       spelling_correct=SPELLING_CORRECT,
                                       replace_nums=REPLACE_NUMS, stem=STEM, remove_stop_words=REMOVE_STOP_WORDS,
                                       remove_punctuation=REMOVE_PUNCTUATION, lower_case=LOWER_CASE,
                                       include_vague=INCLUDE_VAGUE, include_normal=INCLUDE_NORMAL)

# most params below exist ONLY for the purposes of the hashing to and from disk
@memoize_to_disk(filename_prefix=features_filename_prefix)
def extract_features(min_df,
                     rem_infreq, sp_crrct,
                     replace_nos, stem, rem_stop_wds,
                     rem_punc, lcase,
                     win_size, pos_win_size, min_ft_freq,
                     inc_vague, inc_normal,
                     extractors):
    feature_extractor = FeatureExtractorTransformer(extractors)
    return feature_extractor.transform(tagged_essays)

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
pos_tag_window = fact_extract_positional_POS_features((POS_WINDOW_SIZE-1/2))
#TODO - add POS TAGS (positional)
#TODO - add dep parse feats
#extractors = [unigram_window, biigram_window, pos_tag_window]
extractors = [unigram_window, biigram_window]

if pos_tag_window in extractors and STEM:
    raise Exception("POS tagging won't work with stemming on")

essay_feats = extract_features(min_df=MIN_SENTENCE_FREQ, rem_infreq=REMOVE_INFREQUENT,
                               sp_crrct=SPELLING_CORRECT,
                               replace_nos=REPLACE_NUMS, stem=STEM, rem_stop_wds=REMOVE_STOP_WORDS,
                               rem_punc=REMOVE_PUNCTUATION, lcase=LOWER_CASE,
                               win_size=WINDOW_SIZE, pos_win_size=POS_WINDOW_SIZE,
                               min_ft_freq=MIN_FEAT_FREQ, extractors=extractors,
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
wd_train_tags = list(all_tags_above_threshold.union(cause_tags))
#wd_train_tags = regular_tags
#wd_train_tags = regular_tags + cause_tags
#wd_test_tags  = [tag for tag in all_tags if tag.isdigit() or tag == "explicit"]
wd_test_tags  = regular_tags

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

# Linear SVC seems to do better
#fn_create_cls = lambda: LogisticRegression()
fn_create_wd_cls    = lambda : LinearSVC(C=1.0)
fn_create_sent_cls  = lambda : LinearSVC(C=1.0)
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
