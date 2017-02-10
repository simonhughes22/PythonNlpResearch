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

root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder                     = root_folder + "Training/"
test_folder                         = root_folder + "Test/"

train_config = get_config(training_folder)

""" FEATURE EXTRACTION """
train_config["window_size"] = 9
offset = (train_config["window_size"] - 1) / 2

unigram_window_stemmed  = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed  = fact_extract_ngram_features_stemmed(offset, 2)
triigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 3)
unigram_bow_window      = fact_extract_bow_ngram_features(offset, 1)

#optimal CB feature set
extractors = [
    unigram_window_stemmed,
    biigram_window_stemmed,
    triigram_window_stemmed,

    unigram_bow_window,

    extract_dependency_relation,
    extract_brown_cluster
]

# For mongo
extractor_names = map(lambda fn: fn.func_name, extractors)
print("Extractors\n\t" + "\n\t".join(extractor_names))

feat_config = dict(train_config.items() + [("extractors", extractors)])
""" LOAD DATA """
train_tagged_essays = load_process_essays(**train_config)

test_config = dict(train_config.items())
test_config["folder"] = test_folder

test_tagged_essays = load_process_essays(**test_config)
logger.info("Essays loaded- Train: %i Test %i" % (len(train_tagged_essays), len(test_tagged_essays)))

# most params below exist ONLY for the purposes of the hashing to and from disk
train_essay_feats = extract_features(train_tagged_essays, **feat_config)
test_essay_feats  = extract_features(test_tagged_essays,  **feat_config)
logger.info("Features loaded")

""" DEFINE TAGS """
_, lst_all_tags = flatten_to_wordlevel_feat_tags(train_essay_feats)
all_regular_tags = list((t for t in flatten(lst_all_tags) if t[0].isdigit()))

tag_freq = Counter(all_regular_tags )
regular_tags = list(tag_freq.keys())

""" works best with all the pair-wise causal relation codes """
wd_train_tags = regular_tags
wd_test_tags  = regular_tags
# tags to evaluate against

folds = [(train_essay_feats, test_essay_feats)]

""" CLASSIFIERS """
fn_create_wd_cls   = lambda: LogisticRegression() # C=1, dual = False seems optimal
wd_algo   = str(fn_create_wd_cls())
print "Classifier:", wd_algo

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

    wd_td_pred = model.predict(td_X)
    wd_vd_pred = model.predict(vd_X)

    """ TEST Tagger """
    td_wd_predictions_by_code = get_by_code_from_powerset_predictions(wd_td_pred, wd_test_tags)
    vd_wd_predictions_by_code = get_by_code_from_powerset_predictions(wd_vd_pred, wd_test_tags)

    """ Get Actual Ys by code (dict of label to predictions """
    wd_td_ys_by_code = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_by_code = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)
    return td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_by_code, wd_vd_ys_by_code

def evaluate_tagger(dual, C, penalty, fit_intercept, multi_class):
    hyper_opt_params = locals()

    # Gather metrics per fold
    cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

    """ This doesn't run in parallel ! """
    for fold, (essays_TD, essays_VD) in enumerate(folds):
        result = train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags,
                              dual=dual, C=C, penalty=penalty, fit_intercept=fit_intercept,
                              multi_class=multi_class)

        td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag = result
        merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
        merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    # print results for each code
    """ Persist Results to Mongo DB """

    SUFFIX = "_WINDOW_CLASSIFIER_MOST_COMMON_TAG_MULTICLASS"
    CB_TAGGING_TD, CB_TAGGING_VD = "TEST_CB_TAGGING_TD" + SUFFIX, "TEST_CB_TAGGING_VD" + SUFFIX
    parameters = dict(train_config)
    parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
    parameters["min_feat_freq"] = MIN_FEAT_FREQ
    parameters.update(hyper_opt_params)

    wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

    # This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
    avg_f1 = float(processor.get_metric(CB_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1


from traceback import format_exc

for dual in [True]:
    for fit_intercept in [True]:
        for penalty in ["l2"]:
            for C in [0.5]:
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
