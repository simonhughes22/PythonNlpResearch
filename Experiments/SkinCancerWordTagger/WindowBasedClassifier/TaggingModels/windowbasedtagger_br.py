# coding=utf-8
import logging

from sklearn.linear_model import LogisticRegression

import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from IterableFP import flatten
from featureextractionfunctions import *
from featurevectorizer import FeatureVectorizer
from load_data import load_process_essays, extract_features
from results_procesor import ResultsProcessor
from window_based_tagger_config import get_config
from wordtagginghelper import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()

root_folder = settings.data_directory + "SkinCancer/Thesis_Dataset/"
folder =                            root_folder + "Training/"
processed_essay_filename_prefix =   root_folder + "Pickled/essays_proc_pickled_"
features_filename_prefix =          root_folder + "Pickled/feats_pickled_"
config = get_config(folder)

#Doesn't seem to make much of a difference
#config["stem"] = True
#config["lower_case"] = True

""" FEATURE EXTRACTION """
config["window_size"] = 11
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

""" works best with all the pair-wise causal relation codes """
wd_train_tags = regular_tags
wd_test_tags  = regular_tags

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """
fn_create_wd_cls   = lambda: LogisticRegression()
# C=1, dual = False seems optimal

wd_algo   = str(fn_create_wd_cls())

# Gather metrics per fold
cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

folds = cross_validation(essay_feats, CV_FOLDS)

def train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags):
    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    print "\nFold %s" % fold
    print "Training Tagging Model"

    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
    feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
    td_X, vd_X = feature_transformer.fit_transform(td_feats), feature_transformer.transform(vd_feats)

    wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)
    """ TRAIN Tagger """
    tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, fn_create_wd_cls,
                                                    wd_train_tags, verbose=True)
    """ TEST Tagger """
    td_wd_predictions_by_code = test_classifier_per_code(td_X, tag2word_classifier, wd_test_tags)
    vd_wd_predictions_by_code = test_classifier_per_code(vd_X, tag2word_classifier, wd_test_tags)
    return td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag

""" This doesn't run in parallel ! Sequential operation takes exactly same duration """
for fold, (essays_TD, essays_VD) in enumerate(folds):
    result = train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags)
    td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag = result
    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

# print results for each code
logger.info("Training completed")

""" Persist Results to Mongo DB """

SUFFIX = "_WINDOW_CLASSIFIER_BR"
SC_TAGGING_TD, SC_TAGGING_VD = "SC_TAGGING_TD" + SUFFIX, "SC_TAGGING_VD" + SUFFIX
parameters = dict(config)
parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
parameters["min_feat_freq"] = MIN_FEAT_FREQ

wd_td_objectid = processor.persist_results(SC_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(SC_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

# This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs

print processor.results_to_string(wd_td_objectid, SC_TAGGING_TD, wd_vd_objectid, SC_TAGGING_VD, "TAGGING")
logger.info("Results Processed")

