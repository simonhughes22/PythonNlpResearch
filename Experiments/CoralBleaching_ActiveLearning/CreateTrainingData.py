# coding=utf-8

# construct unique key using settings for pickling
import Settings

""" PETER - CHANGE THESE FILE PATHS """
root        = "/Users/simon.hughes/Google Drive/PhD/Data/ActiveLearning/"
data        = root + "EBA1415_Merged/"    # Location where the data is, use EBA_Pre and Post test essays preferably

serialized_features = root + "essay_feats.pl"
serialized_essays   = root + "essays.pl"

""" END SETTINGS """

from featureextractortransformer import FeatureExtractorTransformer
from load_data import load_process_essays
import cPickle as pickle

from featureextractionfunctions import *
from window_based_tagger_config import get_config

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
feature_extractor = FeatureExtractorTransformer(extractors)

essay_feats = feature_extractor.transform(tagged_essays)
logger.info("Features loaded")

pickle.dump(tagged_essays, serialized_essays)
pickle.dump(essay_feats, serialized_features)
logger.info("Serialized")