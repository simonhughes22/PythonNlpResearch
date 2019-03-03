from collections import defaultdict
import pandas as pd
from typing import List, Tuple, Any, Set

from results_procesor import ResultsProcessor
from train_parser import train_sr_parser
from wordtagginghelper import merge_dictionaries

def metrics_to_df(metrics):
    import Rpfa

    rows = []
    for k,val in metrics.items():
        if type(val) == Rpfa.rpfa:
            d = dict(val.__dict__) # convert obj to dict
        elif type(val) == dict:
            d = dict(val)
        else:
            d = dict()
        d["code"] = k
        rows.append(d)
    return pd.DataFrame(rows)

def get_micro_metrics(df):
    return df[df.code == "MICRO_F1"][["accuracy", "f1_score", "recall", "precision"]]

def evaluate_model_essay_level(
        folds: List[Tuple[Any, Any]],
        extractor_fn_names_lst: List[str],
        all_extractor_fns: List[Any],
        beta: float,
        ngrams: int,
        stemmed: bool,
        max_epochs: int,
        min_feat_freq:int,
        cr_tags: Set[str],
        base_learner_fact: Any,
        down_sample_rate=1.0)-> Tuple[Any]:

    if down_sample_rate < 1.0:
        new_folds = []  # type: List[Tuple[Any, Any]]
        for i, (essays_TD, essays_VD) in enumerate(folds):
            essays_TD = essays_TD[:int(down_sample_rate * len(essays_TD))]
            essays_VD = essays_VD[:int(down_sample_rate * len(essays_VD))]
            new_folds.append((essays_TD, essays_VD))
        folds = new_folds  # type: List[Tuple[Any, Any]]

    serial_results = [
        train_sr_parser(essays_TD, essays_VD, extractor_fn_names_lst, all_extractor_fns, ngrams, stemmed, beta,
                        max_epochs, cr_tags, min_feat_freq, cr_tags, base_learner_fact)
        for essays_TD, essays_VD in folds
    ]

    cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

    # record the number of features in each fold
    number_of_feats = []

    # Parallel is almost 5X faster!!!
    parser_models = []
    for (model, num_feats,
         sent_td_ys_bycode, sent_vd_ys_bycode,
         sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode) in serial_results:
        number_of_feats.append(num_feats)

        parser_models.append(model)
        merge_dictionaries(sent_td_ys_bycode, cv_sent_td_ys_by_tag)
        merge_dictionaries(sent_vd_ys_bycode, cv_sent_vd_ys_by_tag)
        merge_dictionaries(sent_td_pred_ys_bycode, cv_sent_td_predictions_by_tag)
        merge_dictionaries(sent_vd_pred_ys_bycode, cv_sent_vd_predictions_by_tag)

    # print(processor.results_to_string(sent_td_objectid, CB_SENT_TD, sent_vd_objectid, CB_SENT_VD, "SENTENCE"))
    return parser_models, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag

def add_cr_labels(observed_tags, ys_bytag_sent, set_cr_tags):
    for tag in set_cr_tags:
        if tag in observed_tags:
            ys_bytag_sent[tag].append(1)
        else:
            ys_bytag_sent[tag].append(0)

def evaluate_ranker(model, xs, essay2crels, ys_bytag, set_cr_tags):
    clone = model.clone()
    if hasattr(model, "average_weights"):
        clone.average_weights()
    pred_ys_bytag = defaultdict(list)
    ename2inps = dict()
    for parser_input in xs:
        ename2inps[parser_input.essay_name] = parser_input

    for ename, act_crels in essay2crels.items():
        if ename not in ename2inps:
            # no predicted crels for this essay
            highest_ranked = set()
        else:
            parser_input = ename2inps[ename]
            ixs = clone.rank(parser_input.all_feats_array)
            highest_ranked = parser_input.all_parses[ixs[0]]  # type: Tuple[str]

        add_cr_labels(set(highest_ranked), pred_ys_bytag, set_cr_tags)

    mean_metrics = ResultsProcessor.compute_mean_metrics(ys_bytag, pred_ys_bytag)
    df = get_micro_metrics(metrics_to_df(mean_metrics))
    return df