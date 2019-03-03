from random import shuffle

from joblib import Parallel, delayed
import numpy as np

from MIRA import CostSensitiveMIRA
from evaluation import evaluate_ranker
from train_parser import get_label_data_essay_level, essay_to_crels

def train_instance(parser_input, model):
    model.train(best_feats=parser_input.opt_features, other_feats_array=parser_input.other_features_array)


def train_cost_sensitive_instance(parser_input, model):
    model.train(best_feats=parser_input.opt_features,
                other_feats_array=parser_input.other_features_array, other_costs_array=parser_input.other_costs_array)


def get_essays_for_data(xs, name2essay):
    return [name2essay[x.essay_name] for x in xs]


def train_model(model, xs_train, xs_test, name2essay, set_cr_tags, max_epochs=30, early_stop_iters=8,
                train_instance_fn=train_instance, verbose=True, early_stopping=True):
    test_accs = [-1]
    best_model = None
    best_test_accuracy = None
    num_declining_acc = 0

    train_essays = get_essays_for_data(xs_train, name2essay)
    test_essays = get_essays_for_data(xs_test,   name2essay)

    ys_by_tag_train = get_label_data_essay_level(train_essays, set_cr_tags)
    ys_by_tag_test = get_label_data_essay_level(test_essays, set_cr_tags)

    essay2crels_train = essay_to_crels(train_essays, set_cr_tags)
    essay2crels_test = essay_to_crels(test_essays,   set_cr_tags)

    xs_train_copy = list(xs_train)
    best_iterations = -1
    for i in range(max_epochs):
        shuffle(xs_train_copy)
        for parser_input in xs_train_copy:
            if len(parser_input.other_parses) > 0:
                train_instance_fn(parser_input, model)

        train_accuracy_df = evaluate_ranker(model, xs_train, essay2crels_train, ys_by_tag_train, set_cr_tags)
        test_accuracy_df = evaluate_ranker(model, xs_test, essay2crels_test, ys_by_tag_test, set_cr_tags)
        train_accuracy = train_accuracy_df.iloc[0].to_dict()["f1_score"]
        test_accuracy = test_accuracy_df.iloc[0].to_dict()["f1_score"]
        if verbose:
            print("Epoch: {epoch} Train Accuracy: {train_acc:.4f} Test Accuracy: {test_acc:.4f}".format(
                epoch=i, train_acc=train_accuracy, test_acc=test_accuracy))

        if not early_stopping or (test_accuracy > max(test_accs)):
            best_model = model.clone()
            best_test_accuracy = test_accuracy_df
            num_declining_acc = 0
            best_iterations = i+1
        else:
            num_declining_acc += 1
            if num_declining_acc >= early_stop_iters:
                break
        test_accs.append(test_accuracy)
    if verbose:
        print("Best Test Acc: {acc:.4f}".format(acc=max(test_accs)))
    return best_model, best_test_accuracy, best_iterations


def shuffle_split_dict(dct, train_pct):
    items = list(dct.items())
    np.random.shuffle(items)
    num_train = int(len(items) * train_pct)
    train_items, test_items = items[:num_train], items[num_train:]
    return dict(train_items), dict(test_items)

def train_model_fold(xs_train, xs_test, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags):
    mdl = CostSensitiveMIRA(C=C, pa_type=pa_type, loss_type=loss_type, max_update_items=max_update_items,
                            initial_weight=1)
    best_mdl, test_acc_df_ml, best_iter = train_model(mdl, xs_train=xs_train, xs_test=xs_test, name2essay=name2essay,
                                           max_epochs=20, early_stop_iters=5, set_cr_tags=set_cr_tags,
                                           train_instance_fn=train_cost_sensitive_instance, verbose=False)
    f1 = test_acc_df_ml["f1_score"].values[0]
    return f1


def train_model_parallel(cv_folds, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags):
    try:
        f1s = Parallel(n_jobs=len(cv_folds))(
            delayed(train_model_fold)(train, test, name2essay, C, pa_type, loss_type, max_update_items, set_cr_tags)
            for (train, test) in cv_folds)
        return np.mean(f1s)
    except KeyboardInterrupt:
        print("Process stopped by user")