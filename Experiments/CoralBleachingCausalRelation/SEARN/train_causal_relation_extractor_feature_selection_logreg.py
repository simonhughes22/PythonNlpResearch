# coding: utf-8
import datetime
import logging
from collections import defaultdict
from typing import Any, List, Set, Tuple

import dill
import numpy as np
import pymongo
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from CrossValidation import cross_validation
from Settings import Settings
from crel_helper import get_cr_tags
from load_data import load_process_essays
from results_procesor import ResultsProcessor, __MICRO_F1__
from function_helpers import get_functions_by_name, get_function_names
from searn_grid_search import make_evaluate_features_closure, get_base_extractor_names, get_all_extractor_names
from searn_parser import SearnModelTemplateFeatures
from template_feature_extractor import NonLocalTemplateFeatureExtractor, NgramExtractor
from window_based_tagger_config import get_config
from wordtagginghelper import merge_dictionaries
from cost_functions import *
from template_feature_extractor import *

# Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Mongo connection
client = pymongo.MongoClient()
db = client.metrics

# Data Set Partition
CV_FOLDS = 5
MIN_FEAT_FREQ = 5

# Global settings

settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder = root_folder + "Training" + "/"
test_folder = root_folder + "Test" + "/"
training_pickled = settings.data_directory + "CoralBleaching/Thesis_Dataset/training.pl"
# NOTE: These predictions are generated from the "./notebooks/SEARN/Keras - Train Tagger and Save CV Predictions For Word Tags.ipynb" notebook
# used as inputs to parsing model
rnn_predictions_folder = root_folder + "Predictions/Bi-LSTM-4-SEARN/"

config = get_config(training_folder)
results_processor = ResultsProcessor(dbname="metrics_causal")

# Get Test Data In Order to Get Test CRELS
# load the test essays to make sure we compute metrics over the test CR labels
test_config = get_config(test_folder)
tagged_essays_test = load_process_essays(**test_config)
########################################################

fname = rnn_predictions_folder + "essays_train_bi_directional-True_hidden_size-256_merge_mode-sum_num_rnns-2_use_pretrained_embedding-True.dill"
with open(fname, "rb") as f:
    pred_tagged_essays = dill.load(f)

logger.info("Started at: " + str(datetime.datetime.now()))
logger.info("Number of pred tagged essays %i" % len(pred_tagged_essays))  # should be 902

cr_tags = get_cr_tags(pr_tagged_essays=pred_tagged_essays, tag_essays_test=tagged_essays_test)
cv_folds = cross_validation(pred_tagged_essays, CV_FOLDS)  # type: List[Tuple[Any,Any]]

# other settings
DOWN_SAMPLE_RATE = 1.0  # For faster smoke testing the algorithm
BETA = 0.2  # ensure hit's zero after 4 tries
MAX_EPOCHS = 10
BASE_LEARNER_FACT = LogisticRegression

# Name of mongo collection
COLLECTION_PREFIX = "CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_FEATURE_SEL"

eval_feats_fn = make_evaluate_features_closure(
    config=config,
    logger=logger,
    results_processor=results_processor,
    collection_prefix=COLLECTION_PREFIX,
    base_learner_fact=BASE_LEARNER_FACT,
    cr_tags=cr_tags,
    max_epochs=MAX_EPOCHS,
    min_feat_freq=MIN_FEAT_FREQ
)

LINE_WIDTH = 80

all_extractor_fn_names = get_all_extractor_names()
base_extractor_fn_names = get_base_extractor_names()

for ngrams in [1]:

    logger.info("*" * LINE_WIDTH)
    logger.info("NGRAM SIZE: {ngram}".format(ngram=ngrams))

    for stemmed in [True, False]:

        logger.info("*" * LINE_WIDTH)
        logger.info("Stemmed: {stemmed}".format(stemmed=stemmed))

        for cost_function_name in [micro_f1_cost_plusepsilon.__name__]:

            logger.info("*" * LINE_WIDTH)
            logger.info("COST FN: {cost_fn}".format(cost_fn=cost_function_name))

            current_extractor_names = []  # type: List[str]
            # current_extractor_names = set(all_extractor_fn_names[1:])
            f1_has_improved = True
            best_f1 = -1.0

            while len(current_extractor_names) <= 5 and \
                  len(current_extractor_names) < len(all_extractor_fn_names) and\
                  f1_has_improved:

                logger.info("-" * LINE_WIDTH)
                logger.info(
                    "Evaluating {num_features} features, with ngram size: {ngrams} and beta decay: {beta_decay}, current feature extractors: {extractors}".format(
                        num_features=len(current_extractor_names) + 1,
                        ngrams=ngrams, beta_decay=BETA,
                        extractors=",".join(current_extractor_names))
                )
                f1_has_improved = False
                # Evaluate new feature sets
                best_new_feature_name = None

                # only use base extractors when no other extractors present as otherwise the more complex features don't kick in
                # as no good parsing decisions can be made
                extractor_names = all_extractor_fn_names if len(
                    current_extractor_names) >= 1 else base_extractor_fn_names
                for new_extractor_name in extractor_names:
                    # Don't add extractors in current set
                    if new_extractor_name in current_extractor_names:
                        continue

                    new_extractor_list = current_extractor_names + [new_extractor_name]  # type : List[str]

                    logger.info("\tExtractors: {extractors}".format(extractors=",".join(new_extractor_list)))
                    # RUN feature evaluation
                    micro_f1 = eval_feats_fn(
                        folds=cv_folds,

                        extractor_fn_names_lst=new_extractor_list,
                        cost_function_name=cost_function_name,
                        ngrams=ngrams,
                        base_learner=LogisticRegression,
                        beta=BETA,
                        stemmed=stemmed,
                        down_sample_rate=DOWN_SAMPLE_RATE)

                    if micro_f1 > best_f1:
                        f1_has_improved = True
                        best_f1 = micro_f1
                        best_new_feature_name = new_extractor_name
                        logger.info(
                            "\t\tMicro F1: {micro_f1} NEW BEST {stars}".format(micro_f1=micro_f1, stars="*" * 30))
                    else:
                        logger.info("\t\tMicro F1: {micro_f1}".format(micro_f1=micro_f1))

                if not f1_has_improved:
                    logger.info("F1 not improved, stopping with {num_extractors} extractors: {extractors}".format(
                        num_extractors=len(current_extractor_names),
                        extractors=",".join(current_extractor_names)
                    ))
                    break
                else:
                    current_extractor_names.append(best_new_feature_name)

## TODO
# - Look into beta decay methods before finalizing - need to determine if this is a good default to use for feat sel
# - Add between tag tags (e.g. explicit tags)

# -TODO - Neat Ideas
# Inject a random action (unform distribution) with a specified probability during training also
# Ensures better exploration of the policy space. Initial algo predictions will be random but converges very quickly so this may be lost
