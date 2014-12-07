__author__ = 'simon.hughes'

import numpy as np
from Decorators import timeit, memoize
from BrattEssay import load_bratt_essays
from processessays import process_sentences, process_essays
from helperfunctions import get_word_feat_tags, get_ys_by_code

from featureextractor import FeatureExtractor
from featuretransformer import FeatureTransformer
from featureextractionfunctions import *
from sklearn.cross_validation import train_test_split
from helperfunctions import *

import pickle
import Settings
import os

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

b4 = set(dir())

MIN_SENTENCE_FREQ   = 0        # i.e. df. Note this is calculated BEFORE creating windows
REMOVE_INFREQUENT   = False    # if false, infrequent words are replaced with "INFREQUENT"
SPELLING_CORRECT    = True
STEM                = False
REPLACE_NUMS        = True     # 1989 -> 0000, 10 -> 00
MIN_SENTENCE_LENGTH = 3
REMOVE_STOP_WORDS   = False
REMOVE_PUNCTUATION  = True
LOWER_CASE          = False
WINDOW_SIZE = 7

PCT_VALIDATION      = 0.2
MIN_FEAT_FREQ       = 15        # 15 best so far, and faster also

after = set(dir())
diff = after - b4
# construct unique key using settings for pickling
pickle_key = "_".join(map(lambda k: k + "_" + str(eval(k)), sorted(filter(lambda a: a != "b4", diff))))
print pickle_key


settings = Settings.Settings()
filename = settings.data_directory + "CoralBleaching/BrattData/pickled_" + pickle_key
if os.path.exists(filename):
    logger.info("Unpickling essays")
    tagged_essays = pickle.load(open(filename))
else:
    # load from disk
    logger.info("Loading Essays")
    # parse bratt essays
    essays = load_bratt_essays()

    logger.info("Processing Essays")
    #TODO pickle this so save time - create a hash key per unique parameter set

    tagged_essays = process_essays(essays,
                                   min_df=MIN_SENTENCE_FREQ,
                                   remove_infrequent=REMOVE_INFREQUENT,
                                   spelling_correct=SPELLING_CORRECT,
                                   replace_nums=REPLACE_NUMS,
                                   stem=STEM,
                                   remove_stop_words=REMOVE_STOP_WORDS,
                                   remove_punctuation=REMOVE_PUNCTUATION,
                                   lower_case=False)

    pickle.dump(tagged_essays, open(filename, "w+"))
    pass

offset = (WINDOW_SIZE-1) / 2
unigram_window = fact_extract_positional_word_features(offset)
biigram_window = fact_extract_ngram_features(offset, 2)
#TODO - add POS TAGS (positional)
#TODO - add dep parse feats
#TODO - memoize features above for speed
extractors = [unigram_window, biigram_window]

feature_extractor = FeatureExtractor(extractors)
essay_feats = feature_extractor.transform(tagged_essays)

""" Data Partitioning and Training """
TD, VD = train_test_split(essay_feats, test_size=PCT_VALIDATION)
td_feats, td_tags = get_word_feat_tags(TD)

feature_transformer = FeatureTransformer(min_feature_freq=MIN_FEAT_FREQ)
td_X = feature_transformer.transform(td_feats)
td_ys_bycode = get_ys_by_code(td_tags)


#TODO call FeatureTransformer and process VD AND TD into xs and ys_per_code

#TODO CV validation loop
#TODO CV split essays, then flatten to word level

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
#   PICKLE THE PRE-PROCESSED ESSAYS:
#       Should also pickle the initial data load such that it's faster when re-running. Use a decorator that hashes the
#       function inputs to a filename in a supplied directory (via the decorator). Spelling correction is time consuming.
#       Don't waste time re-executing.

# OLD
logger.info("Splitting Windows")
winProc = WindowProc(tagged_essays, window_size=7)
windows = winProc.get_word_windows()
ysByCode    = winProc.get_tag_windows()

from WindowFeatureExtractor import  WindowFeatureExtractor as WinExtractor
from WindowFeatures import extract_word_features, extract_positional_word_features, positional_bigram_features

#extract features
extractor = WinExtractor(
    [   extract_word_features,
        extract_positional_word_features,
        positional_bigram_features
    ], MIN_FEAT_FREQ, False)

xs = extractor.fit_transform(windows)

#TODO Include dependency parse features
"""
