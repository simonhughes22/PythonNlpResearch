from collections import defaultdict
from random import shuffle
from typing import Any, Dict, Set

from joblib import Parallel, delayed
import numpy as np

from MIRA import CostSensitiveMIRA
from evaluation import add_cr_labels
from results_procesor import ResultsProcessor, __MICRO_F1__
from train_parser import get_label_data_essay_level, essay_to_crels, List
from wordtagginghelper import merge_dictionaries


def train_instance(parser_input, model):
    model.train(best_feats=parser_input.opt_features, other_feats_array=parser_input.other_features_array)


def train_cost_sensitive_instance(parser_input, model):
    model.train(best_feats=parser_input.opt_features,
                other_feats_array=parser_input.other_features_array, other_costs_array=parser_input.other_costs_array)

def get_essays_for_data(xs, name2essay):
    return [name2essay[x.essay_name] for x in xs]

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
    # df = get_micro_metrics(metrics_to_df(mean_metrics))

    micro_f1 = mean_metrics["MICRO_F1"]["f1_score"]
    return micro_f1, ys_bytag, pred_ys_bytag

def train_model(model, xs_train, xs_test, name2essay, set_cr_tags, max_epochs=30, early_stop_iters=8,
                train_instance_fn=train_instance, verbose=True, early_stopping=True, return_metrics=False):

    test_accs = [-1]
    best_model = None
    best_test_f1 = None
    num_declining_acc = 0

    train_essays = get_essays_for_data(xs_train, name2essay)
    test_essays  = get_essays_for_data(xs_test,  name2essay)

    ys_by_tag_train = get_label_data_essay_level(train_essays, set_cr_tags)
    ys_by_tag_test = get_label_data_essay_level(test_essays, set_cr_tags)

    essay2crels_train = essay_to_crels(train_essays, set_cr_tags)
    essay2crels_test = essay_to_crels(test_essays,   set_cr_tags)

    train_ys_bytag, train_pred_ys_bytag, test_ys_bytag,  test_pred_ys_bytag = None, None, None, None

    xs_train_copy = list(xs_train)
    best_iterations = -1
    for i in range(max_epochs):
        shuffle(xs_train_copy)
        for parser_input in xs_train_copy:
            if len(parser_input.other_parses) > 0:
                train_instance_fn(parser_input, model)

        train_f1, tmp_train_ys_bytag, tmp_train_pred_ys_bytag = evaluate_ranker(model, xs_train, essay2crels_train, ys_by_tag_train, set_cr_tags)
        test_f1,  tmp_test_ys_bytag,  tmp_test_pred_ys_bytag  = evaluate_ranker(model, xs_test, essay2crels_test, ys_by_tag_test, set_cr_tags)

        if verbose:
            print("Epoch: {epoch} Train Accuracy: {train_acc:.4f} Test Accuracy: {test_acc:.4f}".format(
                epoch=i, train_acc=train_f1, test_acc=test_f1))

        if not early_stopping or (test_f1 > max(test_accs)):
            best_model = model.clone()
            num_declining_acc = 0
            best_test_f1 = test_f1
            best_iterations = i+1

            train_ys_bytag, train_pred_ys_bytag, test_ys_bytag, test_pred_ys_bytag = \
                tmp_train_ys_bytag, tmp_train_pred_ys_bytag, tmp_test_ys_bytag,  tmp_test_pred_ys_bytag
        else:
            num_declining_acc += 1
            if num_declining_acc >= early_stop_iters:
                break

        test_accs.append(test_f1)
    if verbose:
        print("Best Test Acc: {acc:.4f}".format(acc=max(test_accs)))

    # If return metrics, then return everything
    if return_metrics:
        return best_test_f1, best_iterations, train_ys_bytag, train_pred_ys_bytag, test_ys_bytag, test_pred_ys_bytag

    return best_model, best_test_f1, best_iterations

def shuffle_split_dict(dct, train_pct):
    items = list(dct.items())
    np.random.shuffle(items)
    num_train = int(len(items) * train_pct)
    train_items, test_items = items[:num_train], items[num_train:]
    return dict(train_items), dict(test_items)

def train_model_fold(xs_train, xs_test, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags,\
                     initial_weight, max_epochs, early_stop_iters):

    mdl = CostSensitiveMIRA(
        C=C, pa_type=pa_type, loss_type=loss_type, max_update_items=max_update_items, initial_weight=initial_weight)

    return train_model(mdl, xs_train=xs_train, xs_test=xs_test, name2essay=name2essay,
            max_epochs=max_epochs, early_stop_iters=early_stop_iters, set_cr_tags=set_cr_tags,
            train_instance_fn=train_cost_sensitive_instance,
            verbose=False, return_metrics=True, early_stopping=False)

def train_model_parallel(cv_folds, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags, \
                         initial_weight, max_epochs=5, early_stop_iters=5, n_jobs=None):

    if n_jobs == None:
        n_jobs = len(cv_folds)
    try:
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_model_fold)(train, test, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags, \
                                      initial_weight, max_epochs, early_stop_iters)
            for (train, test) in cv_folds)

        f1s = []
        for tpl in results:
            best_test_f1, best_iterations, train_ys_bytag, train_pred_ys_bytag, test_ys_bytag, test_pred_ys_bytag = tpl
            f1s.append(best_test_f1)

        return np.mean(f1s)

    except KeyboardInterrupt:
        print("Process stopped by user")


def train_model_parallel_logged(training_collection_name: str, results_processor: ResultsProcessor,
                                feat_extractors: List[str], params: Dict[str, Any],
                                cv_folds: List[Any], name2essay: Dict[str,str],
                                C: float, pa_type: str, loss_type: str, max_update_items:int, set_cr_tags: Set[str], \
                                initial_weight: float,  max_epochs=5, early_stop_iters=5):
    try:
        results = Parallel(n_jobs=len(cv_folds))(
            delayed(train_model_fold)(train, test, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags, \
                                      initial_weight, max_epochs, early_stop_iters)
            for (train, test) in cv_folds)

        cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag = defaultdict(list), defaultdict(list)
        cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

        f1s = []
        for tpl in results:
            best_test_f1, best_iterations, train_ys_bytag, train_pred_ys_bytag, test_ys_bytag, test_pred_ys_bytag = tpl
            f1s.append(best_test_f1)

            merge_dictionaries(train_ys_bytag, cv_sent_td_ys_by_tag)
            merge_dictionaries(test_ys_bytag, cv_sent_vd_ys_by_tag)

            merge_dictionaries(train_pred_ys_bytag, cv_sent_td_predictions_by_tag)
            merge_dictionaries(test_pred_ys_bytag, cv_sent_vd_predictions_by_tag)


        ALGO = "MIRA Cost Sensitive Re-Ranker"
        validation_collection = training_collection_name.replace("_TD", "_VD")

        # extractors = list(map(lambda fn: fn.func_name, feat_extractors))
        extractors = list(feat_extractors)

        parameters = {
            "C": C,
            "pa_type": pa_type,
            "loss_type": loss_type,
            "max_update_items": max_update_items,
            "initial_weight": initial_weight,

            "max_epochs": max_epochs,
            "early_stopping_iters": early_stop_iters,

            "extractors": extractors
        }
        # add in additional parameters not passed in
        parameters.update(params)

        wd_td_objectid = results_processor.persist_results(training_collection_name,
                                                           cv_sent_td_ys_by_tag,
                                                           cv_sent_td_predictions_by_tag,
                                                           parameters, ALGO)

        wd_vd_objectid = results_processor.persist_results(validation_collection,
                                                           cv_sent_vd_ys_by_tag,
                                                           cv_sent_vd_predictions_by_tag,
                                                           parameters, ALGO)

        avg_f1 = float(results_processor.get_metric(validation_collection, wd_vd_objectid, __MICRO_F1__)["f1_score"])
        return avg_f1

    except KeyboardInterrupt:
        print("Process stopped by user")

