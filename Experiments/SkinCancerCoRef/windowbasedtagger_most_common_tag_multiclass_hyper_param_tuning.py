# coding=utf-8
import logging
from collections import Counter

from sklearn.linear_model import LogisticRegression

import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from IterableFP import flatten
from featureextractionfunctions import *
from featurevectorizer import FeatureVectorizer
from load_data import load_process_essays, extract_features
from results_procesor import ResultsProcessor, __MICRO_F1__
from window_based_tagger_config import get_config
from wordtagginghelper import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor(dbname="metrics_coref_new")

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

""" FEATURE EXTRACTION """
config["window_size"] = 5
offset = (config["window_size"] - 1) // 2
#python 3.x
#offset = (config["window_size"] - 1) // 2

unigram_window = fact_extract_positional_word_features(offset)
bigram_window = fact_extract_positional_ngram_features(offset, 2)
trigram_window = fact_extract_positional_ngram_features(offset, 3)
unigram_bow_window = fact_extract_bow_ngram_features(offset, 1)
bigram_bow_window = fact_extract_bow_ngram_features(offset, 2)
trigram_bow_window = fact_extract_bow_ngram_features(offset, 3)

pos_tag_window = fact_extract_positional_POS_features(offset)

# optimal feats from the tuning
extractors = [
        unigram_window,
        bigram_bow_window,
        unigram_bow_window,
        bigram_window,
        pos_tag_window,
        trigram_window
    ]

feat_config = dict(list(config.items()) + [("extractors", extractors)])

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
#all_regular_tags = list((t for t in flatten(lst_all_tags) if t[0].isdigit()))
all_regular_tags = list(set((t for t in flatten(lst_all_tags) if t.lower().strip() == "anaphor" )))
tag_freq = Counter(all_regular_tags)
regular_tags = list(tag_freq.keys())

""" works best with all the pair-wise causal relation codes """
wd_train_tags = regular_tags
wd_test_tags  = regular_tags

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """
fn_create_wd_cls   = lambda: LogisticRegression() # C=1, dual = False seems optimal
wd_algo   = str(fn_create_wd_cls())
print("Classifier:", wd_algo)

folds = cross_validation(essay_feats, CV_FOLDS)

def train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags,
                 dual, C, penalty, fit_intercept, multi_class):

    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
    feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
    td_X, vd_X = feature_transformer.fit_transform(td_feats), feature_transformer.transform(vd_feats)

    """ compute most common tags per word for training only (but not for evaluation) """
    wd_td_ys = get_wordlevel_mostfrequent_ys(td_tags, wd_train_tags, tag_freq)

    """ TRAIN Tagger """
    solver = 'liblinear'
    if multi_class == 'multinomial':
        solver = "lbfgs"
    model = LogisticRegression(dual=dual, C=C, penalty=penalty, fit_intercept=fit_intercept, multi_class=multi_class, solver=solver)
    if fold == 0:
        print(model)

    model.fit(td_X, wd_td_ys)
    num_feats = model.coef_.shape[-1]

    wd_td_pred = model.predict(td_X)
    wd_vd_pred = model.predict(vd_X)

    """ TEST Tagger """
    td_wd_predictions_by_code = get_by_code_from_powerset_predictions(wd_td_pred, wd_test_tags)
    vd_wd_predictions_by_code = get_by_code_from_powerset_predictions(wd_vd_pred, wd_test_tags)

    """ Get Actual Ys by code (dict of label to predictions """
    wd_td_ys_by_code = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_by_code = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)
    return num_feats, td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_by_code, wd_vd_ys_by_code

def evaluate_tagger(dual, C, penalty, fit_intercept, multi_class):
    hyper_opt_params = locals()

    # Gather metrics per fold
    cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

    """ This doesn't run in parallel ! """

    number_of_feats = []
    for fold, (essays_TD, essays_VD) in enumerate(folds):
        result = train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags,
                              dual=dual, C=C, penalty=penalty, fit_intercept=fit_intercept,
                              multi_class=multi_class)

        num_fts, td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag = result
        number_of_feats.append(num_fts)
        merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
        merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    # print results for each code
    """ Persist Results to Mongo DB """

    avg_feats = np.mean(number_of_feats)

    SUFFIX = "_HYPER_PARAM_TUNING"
    CB_TAGGING_TD, CB_TAGGING_VD = "SC_TAGGING_TD" + SUFFIX, "SC_TAGGING_VD" + SUFFIX

    parameters = dict(config)
    parameters["extractors"] = list(map(lambda fn: fn.func_name, extractors))
    parameters["min_feat_freq"] = MIN_FEAT_FREQ
    parameters["num_feats_MEAN"] = avg_feats
    parameters["num_feats_per_fold"] = number_of_feats
    parameters.update(hyper_opt_params)

    wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

    # This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
    avg_f1 = float(processor.get_metric(CB_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1


from traceback import format_exc

for dual in [True, False]:
    #for fit_intercept in [True, False]: # remove as non-optimal and to speed up
    for fit_intercept in [True]:
        for penalty in ["l1", "l2"]:
            # dual only support l2
            if dual and penalty != "l2":
                continue
            for C in [0.1, 0.5, 1.0, 10.0, 100.0]:
                # fof multinomial, we force solver from liblinear to lbfgs, as required by sklearn implementation
                #for multi_class in ['ovr', 'multinomial']: # remove as non-optimal and to speed up
                for multi_class in ['ovr']:
                    try:
                        avg_f1 = evaluate_tagger(dual=dual, C=C, penalty=penalty,
                                                 fit_intercept=fit_intercept, multi_class=multi_class)

                        logger.info("AVG: F1: %s\n\tmulti_class: %s dual: %s penalty: %s fit_intercept: %s C:%s"
                                    % (str(round(avg_f1, 6)).rjust(8),
                                       multi_class.ljust(12), str(dual), str(penalty),
                                       str(fit_intercept), str(round(C, 3)).rjust(5)))
                    except:
                        print(format_exc())
