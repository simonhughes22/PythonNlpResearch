# coding: utf-8
import datetime
import logging
from collections import defaultdict
from typing import Any

import dill
import numpy as np
import pymongo
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from CrossValidation import cross_validation
from Settings import Settings
from cost_functions import *
from crel_helper import get_cr_tags
from function_helpers import get_function_names, get_functions_by_name
from load_data import load_process_essays
from results_procesor import ResultsProcessor, __MICRO_F1__
from searn_essay_parser import SearnModelEssayParser
from searn_parser import SearnModelTemplateFeatures
from template_feature_extractor import *
from window_based_tagger_config import get_config
from wordtagginghelper import merge_dictionaries
from global_template_features import gbl_adjacent_sent_code_features, gbl_causal_features, \
    gbl_concept_code_cnt_features, gbl_ratio_features, gbl_sentence_position_features


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
results_processor = ResultsProcessor(dbname="metrics_causal_model")

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

cr_tags = get_cr_tags(train_tagged_essays=pred_tagged_essays, tag_essays_test=tagged_essays_test)
cv_folds = cross_validation(pred_tagged_essays, CV_FOLDS)  # type: List[Tuple[Any,Any]]

def evaluate_features(
        collection_prefix: str,
        folds: List[Tuple[Any, Any]],
        extractor_fn_names_lst: List[str],
        cost_function_name: str,
        beta: float,
        base_learner: Any,
        ngrams: int,
        stemmed: bool,
        down_sample_rate=1.0) -> float:

    if down_sample_rate < 1.0:
        new_folds = []  # type: List[Tuple[Any, Any]]
        for i, (essays_TD, essays_VD) in enumerate(folds):
            essays_TD = essays_TD[:int(down_sample_rate * len(essays_TD))]
            essays_VD = essays_VD[:int(down_sample_rate * len(essays_VD))]
            new_folds.append((essays_TD, essays_VD))
        folds = new_folds  # type: List[Tuple[Any, Any]]

    parallel_results = Parallel(n_jobs=len(folds))(
        delayed(model_train_predict)(essays_TD, essays_VD, extractor_fn_names_lst, cost_function_name, ngrams, stemmed,
                                     beta)
        for essays_TD, essays_VD in folds)

    cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

    # record the number of features in each fold
    number_of_feats = []

    # Parallel is almost 5X faster!!!
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
    parameters["extractors"] = list(extractor_fn_names_lst)
    parameters["num_extractors"] = len(extractor_fn_names_lst)
    parameters["cost_function"] = cost_function_name
    parameters["beta"] = beta
    parameters["max_epochs"] = MAX_EPOCHS
    parameters["no_stacking"] = True
    parameters["algorithm"] = str(base_learner())
    parameters["ngrams"] = str(ngrams)
    parameters["num_feats_MEAN"] = avg_feats
    parameters["num_feats_per_fold"] = number_of_feats
    parameters["min_feat_freq"] = MIN_FEAT_FREQ
    parameters["stemmed"] = stemmed

    logger.info("\t\tMean num feats: {avg_feats:.2f}".format(avg_feats=avg_feats))

    TD = collection_prefix + "_TD"
    VD = collection_prefix + "_VD"
    if down_sample_rate < 1.0:
        logger.info("\t\tDown sampling at rate: {rate:.5f}, storing temp results".format(rate=down_sample_rate))
        parameters["down_sample"] = down_sample_rate
        CB_SENT_TD, CB_SENT_VD = "__tmp_" + TD, "__tmp_" + TD
    else:
        CB_SENT_TD, CB_SENT_VD = TD, VD

    sent_td_objectid = results_processor.persist_results(CB_SENT_TD, cv_sent_td_ys_by_tag,
                                                         cv_sent_td_predictions_by_tag, parameters, sent_algo)
    sent_vd_objectid = results_processor.persist_results(CB_SENT_VD, cv_sent_vd_ys_by_tag,
                                                         cv_sent_vd_predictions_by_tag, parameters, sent_algo)

    # print(processor.results_to_string(sent_td_objectid, CB_SENT_TD, sent_vd_objectid, CB_SENT_VD, "SENTENCE"))
    micro_f1 = float(results_processor.get_metric(CB_SENT_VD, sent_vd_objectid, __MICRO_F1__)["f1_score"])
    return micro_f1


def model_train_predict(essays_TD, essays_VD, extractor_names, cost_function_name, ngrams, stemmed, beta):
    extractors = get_functions_by_name(extractor_names, all_extractor_fns)
    # get single cost function
    cost_fn = get_functions_by_name([cost_function_name], all_cost_functions)[0]
    assert cost_fn is not None, "Cost function look up failed"
    # Ensure all extractors located
    assert len(extractors) == len(extractor_names), "number of extractor functions does not match the number of names"

    template_feature_extractor = NonLocalTemplateFeatureExtractor(extractors=extractors)
    if stemmed:
        ngram_extractor = NgramExtractorStemmed(max_ngram_len=ngrams)
    else:
        ngram_extractor = NgramExtractor(max_ngram_len=ngrams)

    parse_model = SearnModelEssayParser(feature_extractor=template_feature_extractor,
                                        cost_function=cost_fn,
                                        min_feature_freq=MIN_FEAT_FREQ,
                                        ngram_extractor=ngram_extractor, cr_tags=cr_tags,
                                        base_learner_fact=BASE_LEARNER_FACT,
                                        beta=beta,
                                        # log_fn=lambda s: print(s))
                                        log_fn=lambda s: None)

    parse_model.train(essays_TD, MAX_EPOCHS)

    num_feats = template_feature_extractor.num_features()

    sent_td_ys_bycode = parse_model.get_label_data(essays_TD)
    sent_vd_ys_bycode = parse_model.get_label_data(essays_VD)

    sent_td_pred_ys_bycode = parse_model.predict(essays_TD)
    sent_vd_pred_ys_bycode = parse_model.predict(essays_VD)

    return num_feats, sent_td_ys_bycode, sent_vd_ys_bycode, sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode


LINE_WIDTH = 80

# other settings
DOWN_SAMPLE_RATE = 1.0  # For faster smoke testing the algorithm
BETA = 0.5
MAX_EPOCHS = 2

# Use optimal settngs from CRel extraction exercise
BASE_LEARNER_FACT = lambda: LogisticRegression(
                        dual=True,
                        C=0.5,
                        penalty="l2",
                        fit_intercept=True)

COLLECTION_PREFIX = "CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_FEATURE_SEL"

# some of the other extractors aren't functional if the system isn't able to do a basic parse
# so the base extractors are the MVP for getting to a basic parser, then additional 'meta' parse
# features from all_extractors can be included
base_extractors = [
    single_words,
    word_pairs,
    three_words,
    between_word_features
]
gbl_extractors = [
    gbl_adjacent_sent_code_features,
    gbl_causal_features,
    gbl_concept_code_cnt_features,
    gbl_sentence_position_features,
    gbl_ratio_features
]

third_order_extractors = [
    word_distance,
    valency,
    unigrams,
    third_order,
    label_set,
    size_features
]

all_extractor_fns = base_extractors + third_order_extractors + gbl_extractors

all_cost_functions = [
    micro_f1_cost,
    micro_f1_cost_squared,
    micro_f1_cost_plusone,
    micro_f1_cost_plusepsilon,
    binary_cost,
    inverse_micro_f1_cost,
    uniform_cost
]

all_extractor_fn_names = get_function_names(all_extractor_fns)
gbl_extractor_fn_names = get_function_names(gbl_extractors)
base_extractor_fn_names = get_function_names(base_extractors)
all_cost_fn_names = get_function_names(all_cost_functions)

for ngrams in [1]:

    logger.info("*" * LINE_WIDTH)
    logger.info("NGRAM SIZE: {ngram}".format(ngram=ngrams))

    for stemmed in [True]:

        logger.info("*" * LINE_WIDTH)
        logger.info("Stemmed: {stemmed}".format(stemmed=stemmed))

        # update top level stem setting too
        config["stem"] = stemmed

        for cost_function_name in [micro_f1_cost_plusepsilon.__name__]:

            logger.info("*" * LINE_WIDTH)
            logger.info("COST FN: {cost_fn}".format(cost_fn=cost_function_name))

            # current_extractor_names = []  # type: List[str]
            current_extractor_names = ['single_words', 'between_word_features', 'label_set',
                                    'three_words', 'third_order', 'unigrams']  # type: List[str]

            logger.info("-" * LINE_WIDTH)
            logger.info(
                "Evaluating {num_features} features, with ngram size: {ngrams} and beta decay: {beta_decay}, current feature extractors: {extractors}".format(
                    num_features=len(current_extractor_names) + 1,
                    ngrams=ngrams, beta_decay=BETA,
                    extractors=",".join(current_extractor_names))
            )
            logger.info("\tExtractors: {extractors}".format(extractors=",".join(current_extractor_names)))

            # Try once using existing features
            micro_f1 = evaluate_features(
                collection_prefix=COLLECTION_PREFIX,
                folds=cv_folds,
                extractor_fn_names_lst=current_extractor_names,
                cost_function_name=cost_function_name,
                ngrams=ngrams,
                base_learner=BASE_LEARNER_FACT,
                beta=BETA,
                stemmed=stemmed,
                down_sample_rate=DOWN_SAMPLE_RATE)

            best_f1 = micro_f1
            logger.info("\t\tMicro F1: {micro_f1} NEW BEST {stars}".format(micro_f1=micro_f1, stars="*" * 30))

            remaining_extractor_names = set(gbl_extractor_fn_names)

            while len(remaining_extractor_names) > 0:

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

                for new_extractor_name in sorted(remaining_extractor_names):

                    new_extractor_list = current_extractor_names + [new_extractor_name]  # type : List[str]

                    logger.info("\tExtractors: {extractors}".format(extractors=",".join(new_extractor_list)))
                    # RUN feature evaluation
                    micro_f1 = evaluate_features(
                        collection_prefix=COLLECTION_PREFIX,
                        folds=cv_folds,
                        extractor_fn_names_lst=new_extractor_list,
                        cost_function_name=cost_function_name,
                        ngrams=ngrams,
                        base_learner=BASE_LEARNER_FACT,
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
                    remaining_extractor_names.remove(best_new_feature_name)
