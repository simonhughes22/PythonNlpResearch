# coding=utf-8
from Decorators import memoize_to_disk
from load_data import load_process_essays, extract_features

from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from results_procesor import ResultsProcessor, __MACRO_F1__, __MICRO_F1__
from window_based_tagger_config import get_config

from joblib import Parallel, delayed
# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# END Classifiers

import Settings
import logging
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

folder = settings.data_directory + "SkinCancer/EBA1415_Merged/"
processed_essay_filename_prefix = settings.data_directory + "SkinCancer/Pickled/essays_proc_pickled_"
features_filename_prefix =          settings.data_directory + "SkinCancer/Pickled/feats_pickled_"

config = get_config(folder)

""" Load Essays """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
logger.info("Essays loaded")
""" End load Essays """

def evaluate_window_size(config, window_size, features_filename_prefix):

    config["window_size"] = window_size

    """ FEATURE EXTRACTION """
    offset = (config["window_size"] - 1) / 2

    unigram_window = fact_extract_positional_word_features(offset)
    bigram_window = fact_extract_positional_ngram_features(offset, 2)
    trigram_window = fact_extract_positional_ngram_features(offset, 3)
    unigram_bow_window = fact_extract_bow_ngram_features(offset, 1)
    bigram_bow_window = fact_extract_bow_ngram_features(offset, 2)
    trigram_bow_window = fact_extract_bow_ngram_features(offset, 3)
    unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
    bigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)
    trigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 3)
    pos_tag_window = fact_extract_positional_POS_features(offset)
    bow_pos_tag_bow_window = fact_extract_bow_POS_features(offset)

    base_extractors = [
        unigram_window,
        unigram_bow_window,
        unigram_window_stemmed,

        pos_tag_window,
        bow_pos_tag_bow_window,

        extract_dependency_relation,
        extract_brown_cluster
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

    existing_extractors = []
    best_avg_f1 = 0.0

    while len(existing_extractors) <= 5:
        f1_improved = False
        new_best_feature_set = None
        hs_existing_extractors = set(map(lambda fn: fn.func_name, existing_extractors))

        for feat_extractor in all_extractors:
            if feat_extractor.func_name in hs_existing_extractors:
                continue
            avg_f1 = evaluate_feature_set(config, existing_extractors, feat_extractor, features_filename_prefix)
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                f1_improved = True
                new_best_feature_set = feat_extractor
                print(("*" * 8) + " NEW BEST F1 " + ("*" * 8))
            print("Window Size: " + str(window_size) + "\tFeature Set (" + str(len(existing_extractors)) + "): " + ",".join(hs_existing_extractors) + ","
                  + feat_extractor.func_name.ljust(50) + " attained F1: " + str(avg_f1))
        # end 'for each feat extractor'

        if not f1_improved:
            break
        else:
            existing_extractors.append(new_best_feature_set)
    return best_avg_f1

def evaluate_feature_set(config, existing_extractors, new_extractor, features_filename_prefix):

    feat_extractors = existing_extractors + [new_extractor]
    feat_config = dict(config.items() + [("extractors", feat_extractors)])
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
    cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)
    folds = cross_validation(essay_feats, CV_FOLDS)

    def train_tagger(essays_TD, essays_VD, wd_test_tags, wd_train_tags):
        # TD and VD are lists of Essay objects. The sentences are lists
        # of featureextractortransformer.Word objects
        """ Data Partitioning and Training """
        td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
        vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
        feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
        td_X, vd_X = feature_transformer.fit_transform(td_feats), feature_transformer.transform(vd_feats)
        wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
        wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)
        """ TRAIN Tagger """
        tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, lambda: LogisticRegression(),
                                                        wd_train_tags, verbose=False)
        """ TEST Tagger """
        td_wd_predictions_by_code = test_classifier_per_code(td_X, tag2word_classifier, wd_test_tags)
        vd_wd_predictions_by_code = test_classifier_per_code(vd_X, tag2word_classifier, wd_test_tags)
        return td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag

    #results = Parallel(n_jobs=CV_FOLDS)(
    #        delayed(train_tagger)(essays_TD, essays_VD, wd_test_tags, wd_train_tags)
    #            for (essays_TD, essays_VD) in folds)

    results = [train_tagger(essays_TD, essays_VD, wd_test_tags, wd_train_tags)
               for (essays_TD, essays_VD) in folds]

    for result in results:
        td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag = result
        merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
        merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    # print results for each code
    """ Persist Results to Mongo DB """
    SUFFIX = "_FEAT_SELECTION"
    SC_TAGGING_TD, SC_TAGGING_VD = "SC_TAGGING_TD" + SUFFIX, "SC_TAGGING_VD" + SUFFIX
    parameters = dict(config)
    parameters["extractors"] = map(lambda fn: fn.func_name, feat_extractors)
    parameters["min_feat_freq"] = MIN_FEAT_FREQ

    wd_td_objectid = processor.persist_results(SC_TAGGING_TD, cv_wd_td_ys_by_tag,
                                               cv_wd_td_predictions_by_tag, parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(SC_TAGGING_VD, cv_wd_vd_ys_by_tag,
                                               cv_wd_vd_predictions_by_tag, parameters, wd_algo)

    avg_f1 = float(processor.get_metric(SC_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1


""" FIND BEST WINDOW SIZE """

best_win_size = -1
best_micro_f1 = 0
#for win_size in [1, 7, 3, 5, 9]:
#for win_size in [9, 11, 13]:
for win_size in [15]:
    macro_f1 = evaluate_window_size(config=config, window_size=win_size, features_filename_prefix=features_filename_prefix)
    if macro_f1 > best_micro_f1:
        print(("!" * 8) + " NEW BEST AVERAGE F1 FOR WINDOW SIZE " + ("!" * 8))
    print("Best average F1 for window size: " + str(win_size) + " is " + str(macro_f1))


