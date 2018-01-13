from typing import List, Any, Tuple

from collections import defaultdict
from joblib import Parallel, delayed

from function_helpers import get_functions_by_name, get_function_names
from searn_parser import SearnModelTemplateFeatures
from template_feature_extractor import NgramExtractor, NgramExtractorStemmed, NonLocalTemplateFeatureExtractor
from wordtagginghelper import merge_dictionaries
import numpy as np
from cost_functions import *
from template_feature_extractor import *
from results_procesor import __MICRO_F1__

# some of the other extractors aren't functional if the system isn't able to do a basic parse
# so the base extractors are the MVP for getting to a basic parser, then additional 'meta' parse
# features from all_extractors can be included
base_extractors = [
    single_words,
    word_pairs,
    three_words,
    between_word_features
]

all_extractor_fns = base_extractors + [
    word_distance,
    valency,
    unigrams,
    third_order,
    label_set,
    size_features
]

all_cost_functions = [
    micro_f1_cost,
    micro_f1_cost_squared,
    micro_f1_cost_plusone,
    micro_f1_cost_plusepsilon,
    binary_cost,
    inverse_micro_f1_cost,
    uniform_cost
]

__all_extractor_fn_names__ = get_function_names(all_extractor_fns)
__base_extractor_fn_names__ = get_function_names(base_extractors)

def get_all_extractor_names():
    return __all_extractor_fn_names__

def get_base_extractor_names():
    return __base_extractor_fn_names__

def make_evaluate_features_closure(config, logger, results_processor, collection_prefix, base_learner_fact, cr_tags, max_epochs, min_feat_freq):

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
        parse_model = SearnModelTemplateFeatures(feature_extractor=template_feature_extractor,
                                                 cost_function=cost_fn,
                                                 min_feature_freq=min_feat_freq,
                                                 ngram_extractor=ngram_extractor, cr_tags=cr_tags,
                                                 base_learner_fact=base_learner_fact,
                                                 beta=beta,
                                                 # log_fn=lambda s: print(s))
                                                 log_fn=lambda s: None)

        parse_model.train(essays_TD, max_epochs)

        num_feats = template_feature_extractor.num_features()

        sent_td_ys_bycode = parse_model.get_label_data(essays_TD)
        sent_vd_ys_bycode = parse_model.get_label_data(essays_VD)

        sent_td_pred_ys_bycode = parse_model.predict(essays_TD)
        sent_vd_pred_ys_bycode = parse_model.predict(essays_VD)

        return num_feats, sent_td_ys_bycode, sent_vd_ys_bycode, sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode

    def evaluate_features(
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
        parameters["max_epochs"] = max_epochs
        parameters["no_stacking"] = True
        parameters["algorithm"] = str(base_learner())
        parameters["ngrams"] = str(ngrams)
        parameters["num_feats_MEAN"] = avg_feats
        parameters["num_feats_per_fold"] = number_of_feats
        parameters["min_feat_freq"] = min_feat_freq
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

    return evaluate_features
