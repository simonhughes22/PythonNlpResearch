# coding=utf-8

# construct unique key using settings for pickling


""" PETER - CHANGE THESE FILE PATHS """
""" DATA - raw essay files + annotations"""
root        = "/Users/simon.hughes/Google Drive/PhD/Data/GlobalWarming/BrattFiles/"
data        = root + "globwarm20new/"    # Location where the data is

""" OUTPUT - two serialized files, one for the pre-processed essays, the other for the features """
serialized_features = data + "Experiment/essay_feats.pl"
serialized_essays   = data + "Experiment/essays.pl"
""" END SETTINGS """

from featureextractortransformer import FeatureExtractorTransformer
from load_data import load_process_essays

from featureextractionfunctions import *
from window_based_tagger_config import get_config

import cPickle as pickle
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# not hashed as don't affect persistence of feature processing

config = get_config(data)

""" FEATURE EXTRACTION """
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

extractors = [unigram_window_stemmed, biigram_window_stemmed]
feat_config = dict(config.items() + [("extractors", extractors)])

""" LOAD DATA """
tagged_essays = load_process_essays( **config )
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk

# Collapse all variants of a tag into one tag
feature_extractor = FeatureExtractorTransformer(extractors)

essay_feats = feature_extractor.transform(tagged_essays)
logger.info("Features loaded")

with open(serialized_essays, "w+") as f_essays:
    pickle.dump(tagged_essays, f_essays)

with open(serialized_features, "w+") as f_feats:
    pickle.dump(essay_feats,   f_feats)

logger.info("Serialized")