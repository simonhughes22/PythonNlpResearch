# coding: utf-8
import datetime
import logging
import dill
import pymongo
import numpy as np
from joblib import Parallel, delayed

from collections import defaultdict
from typing import Any, List, Set, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from CrossValidation import cross_validation
from Settings import Settings
from load_data import load_process_essays
from results_procesor import ResultsProcessor, __MICRO_F1__
from searn_parser_template_features import SearnModelTemplateFeaturesCostSensitive
from template_feature_extractor import NonLocalTemplateFeatureExtractor, NgramExtractor
from template_feature_extractor import single_words, word_pairs, three_words, word_distance, valency, unigrams, \
    between_word_features
from window_based_tagger_config import get_config
from wordtagginghelper import merge_dictionaries

# Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Mongo connection
client = pymongo.MongoClient()
db = client.metrics

# Data Set Partition
CV_FOLDS = 5
DEV_SPLIT = 0.1

# Global settings
MAX_EPOCHS = 12

settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder = root_folder + "Training" + "/"
test_folder = root_folder + "Test" + "/"
training_pickled = settings.data_directory + "CoralBleaching/Thesis_Dataset/training.pl"
# NOTE: These predictions are generated from the "./notebooks/SEARN/Keras - Train Tagger and Save CV Predictions For Word Tags.ipynb" notebook
# used as inputs to parsing model
rnn_predictions_folder = root_folder + "Predictions/Bi-LSTM-4-SEARN/"

config = get_config(training_folder)
processor = ResultsProcessor()

# Get Test Data In Order to Get Test CRELS
# load the test essays to make sure we compute metrics over the test CR labels
test_config = get_config(test_folder)
tagged_essays_test = load_process_essays(**test_config)
########################################################

fname = rnn_predictions_folder + "essays_train_bi_directional-True_hidden_size-256_merge_mode-sum_num_rnns-2_use_pretrained_embedding-True.dill"
with open(fname, "rb") as f:
    pred_tagged_essays = dill.load(f)

logger.info("Started at: " + str(datetime.datetime.now()))
logger.info("Number of pred tagged essays %i" % len(pred_tagged_essays) ) # should be 902

# In[7]:

CAUSER = "Causer"
RESULT = "Result"
EXPLICIT = "explicit"
CAUSER_EXPLICIT = "Causer_Explicit"
EXPLICIT_RESULT = "Explicit_Result"
CAUSER_EXPLICIT_RESULT = "Causer_Explicit_Result"
CAUSER_RESULT = "Causer_Result"

stag_freq = defaultdict(int)
unique_words = set()
for essay in pred_tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            unique_words.add(word)
            for tag in tags:
                stag_freq[tag] += 1

for essay in tagged_essays_test:
    for sentence in essay.sentences:
        for word, tags in sentence:
            unique_words.add(word)
            for tag in tags:
                stag_freq[tag] += 1

# TODO - don't ignore Anaphor, other and rhetoricals here
cr_tags = list((t for t in stag_freq.keys() if ( "->" in t) and
                not "Anaphor" in t and
                not "other" in t and
                not "rhetorical" in t and
                not "factor" in t and
                1==1
               ))

#Change to include explicit
regular_tags = set((t for t in stag_freq.keys() if ( "->" not in t) and (t == "explicit" or t[0].isdigit())))
vtags = set(regular_tags)

assert "explicit" in vtags, "explicit should be in the regular tags"

cv_folds = cross_validation(pred_tagged_essays, CV_FOLDS) # type: List[Tuple[Any,Any]]

# some of the other extractors aren't functional if the system isn't able to do a basic parse
# so the base extractors are the MVP for getting to a basic parser, then additional 'meta' parse
# features from all_extractors can be included
base_extractors = [
    single_words, word_pairs, three_words, between_word_features
]

all_extractors = base_extractors + [
    word_distance,
    valency, unigrams
]

def evaluate_features(  folds : List[Tuple[Any, Any]],
                        extractor_names: Set[str],
                        beta_decay: float = 0.3,
                        base_learner: Any=LogisticRegression,
                        ngrams:int=2, down_sample_rate = 1.0)->float:

    #extractors = [fn for fn in all_extractors if fn.__name__ in extractor_names]
    # Ensure all extractors located
    #assert len(extractors) == len(extractor_names), "number of extractor functions does not match the number of names"

    #TODO - try to parallelize - pass in an array of func names (strings) - and look up real functions due to joblib limitations
    # - worth the effort? This completes in 5 hours with 0.25 Beta decay rate

    if down_sample_rate < 1.0:
        new_folds = [] # type: List[Tuple[Any, Any]]
        for i, (essays_TD, essays_VD) in enumerate(folds):
            essays_TD = essays_TD[:int(down_sample_rate * len(essays_TD))]
            essays_VD = essays_VD[:int(down_sample_rate * len(essays_VD))]
            new_folds.append((essays_TD, essays_VD))
        folds = new_folds # type: List[Tuple[Any, Any]]

    parallel_results = Parallel(n_jobs=len(folds))(
        delayed(model_train_predict)(essays_TD, essays_VD, extractor_names, ngrams, beta_decay)
        for essays_TD, essays_VD in folds)

    cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

    # record the number of features in each fold
    number_of_feats = []

    for (num_feats,
         sent_td_ys_bycode, sent_vd_ys_bycode,
         sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode) in parallel_results:

        number_of_feats.append(num_feats)

        merge_dictionaries(sent_td_ys_bycode, cv_sent_td_ys_by_tag)
        merge_dictionaries(sent_vd_ys_bycode, cv_sent_vd_ys_by_tag)
        merge_dictionaries(sent_td_pred_ys_bycode, cv_sent_td_predictions_by_tag)
        merge_dictionaries(sent_vd_pred_ys_bycode, cv_sent_vd_predictions_by_tag)
        # break

    # Mongo settings recording
    avg_feats = np.mean(number_of_feats)
    sent_algo = "Shift_Reduce_Parser_LR"

    parameters = dict(config)
    parameters["extractors"] = list(sorted(extractor_names))
    parameters["beta_decay"] = beta_decay
    parameters["no_stacking"] = True
    parameters["algorithm"] = str(base_learner())
    parameters["ngrams"] = str(ngrams)
    parameters["num_feats_MEAN"] = avg_feats
    parameters["num_feats_per_fold"] = number_of_feats

    logger.info("\t\tMean num feats: {avg_feats:.2f}".format(avg_feats=avg_feats))

    if down_sample_rate < 1.0:
        logger.info("\t\tDown sampling at rate: {rate:.5f}, storing temp results".format(rate=down_sample_rate))
        parameters["down_sample"] = down_sample_rate
        CB_SENT_TD, CB_SENT_VD = "__tmp_CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_FEATURE_SEL_TD", "__tmp_CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_FEATURE_SEL_VD"
    else:
        CB_SENT_TD, CB_SENT_VD = "CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_FEATURE_SEL_TD", "CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_FEATURE_SEL_VD"

    sent_td_objectid = processor.persist_results(CB_SENT_TD, cv_sent_td_ys_by_tag,
                                                 cv_sent_td_predictions_by_tag, parameters, sent_algo)
    sent_vd_objectid = processor.persist_results(CB_SENT_VD, cv_sent_vd_ys_by_tag,
                                                 cv_sent_vd_predictions_by_tag, parameters, sent_algo)

    #print(processor.results_to_string(sent_td_objectid, CB_SENT_TD, sent_vd_objectid, CB_SENT_VD, "SENTENCE"))
    micro_f1 = float(processor.get_metric(CB_SENT_VD, sent_vd_objectid, __MICRO_F1__)["f1_score"])
    return micro_f1


def model_train_predict(essays_TD, essays_VD, extractor_names, ngrams, beta_decay):

    extractors = [fn for fn in all_extractors if fn.__name__ in extractor_names]
    # Ensure all extractors located
    assert len(extractors) == len(extractor_names), "number of extractor functions does not match the number of names"

    template_feature_extractor = NonLocalTemplateFeatureExtractor(extractors=extractors)
    ngram_extractor = NgramExtractor(max_ngram_len=ngrams)
    parse_model = SearnModelTemplateFeaturesCostSensitive(feature_extractor=template_feature_extractor,
                                                          ngram_extractor=ngram_extractor, cr_tags=cr_tags,
                                                          base_learner_fact=BASE_LEARNER_FACT,
                                                          beta_decay_fn=lambda beta: beta - beta_decay,
                                                          # silent
                                                          log_fn=lambda s: None)
    parse_model.train(essays_TD, MAX_EPOCHS)

    num_feats  =template_feature_extractor.num_features()

    sent_td_ys_bycode = parse_model.get_label_data(essays_TD)
    sent_vd_ys_bycode = parse_model.get_label_data(essays_VD)

    sent_td_pred_ys_bycode = parse_model.predict(essays_TD)
    sent_vd_pred_ys_bycode = parse_model.predict(essays_VD)

    return num_feats, sent_td_ys_bycode, sent_vd_ys_bycode, sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode


def get_extractor_names(feature_extractors):
    return list(map(lambda fn: fn.__name__, feature_extractors))

LINE_WIDTH = 80

all_extractor_names = get_extractor_names(all_extractors)
#all_extractor_names = get_extractor_names([valency])
base_extractor_names = get_extractor_names(base_extractors)

# other settings
DOWN_SAMPLE_RATE    = 1.0  # For faster smoke testing the algorithm
BETA_DECAY          = 0.250001 # ensure hit's zero after 4 tries
BASE_LEARNER_FACT   = LogisticRegression

#TODO - stem words or not?
for ngrams in [2,3,1]:

    current_extractor_names = set()
    f1_improved = True
    best_f1 = -1.0

    while len(current_extractor_names) <= 5 and f1_improved:

        logger.info("-" * LINE_WIDTH)
        logger.info("Evaluating {num_features} features, with ngram size: {ngrams} and beta decay: {beta_decay}, current feature extractors: {extractors}".format(
                        num_features=len(current_extractor_names) + 1,
                        ngrams=ngrams, beta_decay=BETA_DECAY,
                        extractors=",".join(sorted(current_extractor_names)))
        )
        f1_improved = False
        # Evaluate new feature sets
        best_new_feature_name = None

        # only use base extractors when no other extractors present as otherwise the more complex features don't kick in
        # as no good parsing decisions can be made
        extractor_names = all_extractor_names if len(current_extractor_names) >= 1 else base_extractor_names
        for new_extractor_name in extractor_names:
            # Don't add extractors in current set
            if new_extractor_name in current_extractor_names:
                continue

            new_extractor_set = current_extractor_names.union([new_extractor_name])

            logger.info("\tExtractors: {extractors}".format(extractors=",".join(sorted(new_extractor_set))))
            # RUN feature evaluation
            micro_f1 = evaluate_features(folds=cv_folds,
                                         extractor_names=new_extractor_set,
                                         ngrams=ngrams,
                                         base_learner=LogisticRegression,
                                         beta_decay=BETA_DECAY,
                                         down_sample_rate=DOWN_SAMPLE_RATE)
            if micro_f1 > best_f1:
                f1_improved = True
                best_f1 = micro_f1
                best_new_feature_name = new_extractor_name
                logger.info("\t\tMicro F1: {micro_f1} NEW BEST {stars}".format(micro_f1=micro_f1, stars="*" * 30))
            else:
                logger.info("\t\tMicro F1: {micro_f1}".format(micro_f1=micro_f1))

        if not f1_improved:
            logger.info("F1 not improved, stopping with {num_extractors} extractors: {extractors}".format(
                num_extractors=len(current_extractor_names),
                extractors=",".join(sorted(current_extractor_names))
            ))
            break
        else:
            current_extractor_names.add(best_new_feature_name)

## TODO
#- Need to handle relations where same code -> same code

#-TODO - Neat Ideas
# Inject a random action (unform distribution) with a specified probability during training also
# Ensures better exploration of the policy space. Initial algo predictions will be random but converges very quickly so this may be lost
