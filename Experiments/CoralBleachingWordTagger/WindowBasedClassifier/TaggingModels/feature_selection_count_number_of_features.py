# coding=utf-8
import logging

# Classifiers
from sklearn.linear_model import LogisticRegression

# END Classifiers
import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from featureextractionfunctions import *
from featurevectorizer import FeatureVectorizer
from load_data import load_process_essays, extract_features
from window_based_tagger_config import get_config
from wordtagginghelper import *

import sys
if sys.version_info[0] >= 3:
    raise Exception("Does not work in Python 3.x - issues with SpaCy code")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized
#processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()

root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
folder =                            root_folder + "Training/"
processed_essay_filename_prefix =   root_folder + "Pickled/essays_proc_pickled_"
features_filename_prefix =          root_folder + "Pickled/feats_pickled_"

config = get_config(folder)

""" Load Essays """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
logger.info("Essays loaded")
""" End load Essays """

def evaluate_window_size(config, window_size, features_filename_prefix):

    config["window_size"] = window_size

    """ FEATURE EXTRACTION """
    offset = (config["window_size"] - 1) // 2

    unigram_window          = fact_extract_positional_word_features(offset)
    bigram_window           = fact_extract_positional_ngram_features(offset, 2)
    trigram_window          = fact_extract_positional_ngram_features(offset, 3)
    unigram_bow_window      = fact_extract_bow_ngram_features(offset, 1)
    bigram_bow_window       = fact_extract_bow_ngram_features(offset, 2)
    trigram_bow_window      = fact_extract_bow_ngram_features(offset, 3)
    unigram_window_stemmed  = fact_extract_positional_word_features_stemmed(offset)
    bigram_window_stemmed   = fact_extract_ngram_features_stemmed(offset, 2)
    trigram_window_stemmed  = fact_extract_ngram_features_stemmed(offset, 3)
    pos_tag_window          = fact_extract_positional_POS_features(offset)
    bow_pos_tag_bow_window  = fact_extract_bow_POS_features(offset)

    # use factory functions to ensure func_name is attached
    extract_dependency_reln = fact_extract_dependency_relation()
    extract_brown_cl = fact_extract_brown_cluster()

    # Ensure function name attached
    base_extractors = [
        extract_dependency_reln,
        extract_brown_cl,

        unigram_window,
        unigram_bow_window,
        unigram_window_stemmed,

        pos_tag_window,
        bow_pos_tag_bow_window
    ]

    bigram_extractors = [
        bigram_window,
        bigram_bow_window,
        bigram_window_stemmed,
    ]

    trigram_extractors = [
        trigram_window,
        trigram_bow_window,
        trigram_window_stemmed,
    ]

    all_extractors = base_extractors
    if window_size >= 2:
        all_extractors += bigram_extractors

    if window_size >= 3:
        all_extractors += trigram_extractors

    # Extract only one set of features

    for feat_extractor in all_extractors:
        existing_extractors = []
        td_avg_col_size, vd_avg_col_size = evaluate_feature_set(config, existing_extractors, feat_extractor, features_filename_prefix)

        #NOTE: func_name works here as we explicitly attach that property when creating each feature
        #print("Win Size: {win_size}\tFeats: {feat_name} TD_Feats:{td_feats:.4f}\tVD_Feats:{vd_feats:.4f}".format(
        print("Win Size: {win_size}\tFeats: {feat_name} TD_Feats:{td_feats}\tVD_Feats:{vd_feats}".format(
            win_size=window_size, feat_name=feat_extractor.func_name.ljust(50),
            td_feats=td_avg_col_size, vd_feats=vd_avg_col_size
        ))

def evaluate_feature_set(config, existing_extractors, new_extractor, features_filename_prefix):

    feat_extractors = existing_extractors + [new_extractor]
    feat_config = dict(list(config.items()) + [("extractors", feat_extractors)])
    """ LOAD FEATURES """
    # most params below exist ONLY for the purposes of the hashing to and from disk
    #mem_extract_features = memoize_to_disk(filename_prefix=features_filename_prefix, verbose=False)(extract_features)
    #essay_feats = mem_extract_features(tagged_essays, **feat_config)
    essay_feats = extract_features(tagged_essays, **feat_config)
    """ DEFINE TAGS """
    _, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
    regular_tags = list(set((t for t in flatten(lst_all_tags) if t[0].isdigit())))
    """ works best with all the pair-wise causal relation codes """
    wd_train_tags = regular_tags
    wd_test_tags = regular_tags
    """ CLASSIFIERS """
    fn_create_wd_cls = lambda: LogisticRegression()  # C=1, dual = False seems optimal
    wd_algo = str(fn_create_wd_cls())

    # Gather metrics per fold
    folds = cross_validation(essay_feats, CV_FOLDS)

    def train_tagger(essays_TD, essays_VD, wd_test_tags, wd_train_tags):
        # TD and VD are lists of Essay objects. The sentences are lists
        # of featureextractortransformer.Word objects
        """ Data Partitioning and Training """
        td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
        vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
        feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
        td_X, vd_X = feature_transformer.fit_transform(td_feats), feature_transformer.transform(vd_feats)
        return td_X.shape, vd_X.shape

    #results = Parallel(n_jobs=CV_FOLDS)(
    #        delayed(train_tagger)(essays_TD, essays_VD, wd_test_tags, wd_train_tags)
    #            for (essays_TD, essays_VD) in folds)

    td_col_sizes, vd_col_sizes = [], []
    for (essays_TD, essays_VD) in folds:
        td_x_shape, vd_x_shape = train_tagger(essays_TD, essays_VD, wd_test_tags, wd_train_tags)
        td_col_sizes.append(td_x_shape[1])
        vd_col_sizes.append(vd_x_shape[1])
    return np.mean(td_col_sizes), np.mean(vd_col_sizes)

""" FIND BEST WINDOW SIZE """

best_win_size = -1
best_micro_f1 = 0
for win_size in [9]:
    evaluate_window_size(config=config, window_size=win_size, features_filename_prefix=features_filename_prefix)

"""

Win Size: 9	Feats: extract_dependency_relation_internal[]             TD_Feats:4848.0	VD_Feats:4848.0
Win Size: 9	Feats: extract_brown_cluster_internal[]                   TD_Feats:512.8	VD_Feats:512.8
Win Size: 9	Feats: fn_pos_wd_feats[offset:4]                          TD_Feats:8305.0	VD_Feats:8305.0
Win Size: 9	Feats: fn_bow_ngram_feat[ngram_size:1 offset:4]           TD_Feats:1617.8	VD_Feats:1617.8
Win Size: 9	Feats: fn_pos_wd_feats_stemmed[offset:4]                  TD_Feats:6486.6	VD_Feats:6486.6
Win Size: 9	Feats: fn_pos_POS_feats[offset:4]                         TD_Feats:321.8	VD_Feats:321.8
Win Size: 9	Feats: fn_bow_POS_feats[offset:4]                         TD_Feats:40.6	    VD_Feats:40.6
Win Size: 9	Feats: fn_pos_ngram_feat[ngram_size:2 offset:4]           TD_Feats:23740.4	VD_Feats:23740.4
Win Size: 9	Feats: fn_bow_ngram_feat[ngram_size:2 offset:4]           TD_Feats:20957.4	VD_Feats:20957.4
Win Size: 9	Feats: fn_pos_ngram_feat_stemmed[ngram_size:2 offset:4]   TD_Feats:23263.6	VD_Feats:23263.6
Win Size: 9	Feats: fn_pos_ngram_feat[ngram_size:3 offset:4]           TD_Feats:18619.8	VD_Feats:18619.8
Win Size: 9	Feats: fn_bow_ngram_feat[ngram_size:3 offset:4]           TD_Feats:46318.2	VD_Feats:46318.2
Win Size: 9	Feats: fn_pos_ngram_feat_stemmed[ngram_size:3 offset:4]   TD_Feats:18941.6	VD_Feats:18941.6


"""