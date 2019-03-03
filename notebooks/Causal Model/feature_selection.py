from typing import Any

import dill
from sklearn.linear_model import LogisticRegression
import numpy as np

from CrossValidation import cross_validation
from MIRA import CostSensitiveMIRA
from Settings import Settings

from crel_helper import get_cr_tags
from crel_processing import essay_to_crels_cv
from evaluation import evaluate_model_essay_level, get_micro_metrics, metrics_to_df
from feature_extraction import get_features_from_probabilities
from feature_normalization import min_max_normalize_feats
from function_helpers import get_function_names
from results_procesor import ResultsProcessor
from train_parser import essay_to_crels, create_extractor_functions
from cost_functions import micro_f1_cost_plusepsilon
from train_reranker import train_model_parallel, train_model, train_cost_sensitive_instance
from window_based_tagger_config import get_config

# Data Set Partition
CV_FOLDS = 5
MIN_FEAT_FREQ = 5

# Global settings
settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder = root_folder + "Training" + "/"
test_folder = root_folder + "Test" + "/"

coref_root = root_folder + "CoReference/"
coref_output_folder = coref_root + "CRel/"

config = get_config(training_folder)

train_fname = coref_output_folder + "training_crel_anatagged_essays_most_recent_code.dill"
with open(train_fname, "rb") as f:
    pred_tagged_essays_train = dill.load(f)

test_fname = coref_output_folder + "test_crel_anatagged_essays_most_recent_code.dill"
with open(test_fname, "rb") as f:
    pred_tagged_essays_test = dill.load(f)

print(len(pred_tagged_essays_train), len(pred_tagged_essays_test))

cr_tags = get_cr_tags(train_tagged_essays=pred_tagged_essays_train, tag_essays_test=pred_tagged_essays_test)
print(cr_tags[0:10])

set_cr_tags = set(cr_tags)

# other settings
base_extractors, all_extractor_fns, all_cost_functions = create_extractor_functions()

all_extractor_fn_names = get_function_names(all_extractor_fns)
base_extractor_fn_names = get_function_names(base_extractors)
all_cost_fn_names = get_function_names(all_cost_functions)

ngrams = 1
stemmed = True
cost_function_name = micro_f1_cost_plusepsilon.__name__
dual = True
fit_intercept = True
beta = 0.5
max_epochs = 2
C = 0.5
penalty = "l2"

# Note these also differ for SC dataset
BASE_LEARNER_FACT = lambda: LogisticRegression(dual=dual, C=C, penalty=penalty, fit_intercept=fit_intercept)
best_extractor_names = ['single_words', 'between_word_features', 'label_set',
                        'three_words', 'third_order', 'unigrams']  # type: List[str]

test_folds = [(pred_tagged_essays_train, pred_tagged_essays_test)]  # type: List[Tuple[Any,Any]]
cv_folds = cross_validation(pred_tagged_essays_train, CV_FOLDS)  # type: List[Tuple[Any,Any]]

result_test_essay_level = evaluate_model_essay_level(
    folds=cv_folds,
    extractor_fn_names_lst=best_extractor_names,
    all_extractor_fns=all_extractor_fns,
    ngrams=ngrams,
    beta=beta,
    stemmed=stemmed,
    down_sample_rate=1.0,
    max_epochs=max_epochs)

models, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, cv_td_preds_by_sent, cv_sent_vd_ys_by_tag = result_test_essay_level

mean_metrics = ResultsProcessor.compute_mean_metrics(cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag)
print(get_micro_metrics(metrics_to_df(mean_metrics)))

models, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = result_test_essay_level

mean_metrics = ResultsProcessor.compute_mean_metrics(cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag)
print(get_micro_metrics(metrics_to_df(mean_metrics)))

result_final_test = evaluate_model_essay_level(
    folds=test_folds,
    extractor_fn_names_lst=best_extractor_names,
    all_extractor_fns=all_extractor_fns,
    ngrams=ngrams,
    beta=beta,
    stemmed=stemmed,
    down_sample_rate=1.0,
    max_epochs=max_epochs)

models_test, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = result_final_test

mean_metrics = ResultsProcessor.compute_mean_metrics(cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag)
print(get_micro_metrics(metrics_to_df(mean_metrics)))

final_test_model = models_test

all_essays = pred_tagged_essays_train + pred_tagged_essays_test
name2essay = {}
for essay in all_essays:
    name2essay[essay.name] = essay

name2crels = essay_to_crels(all_essays, set_cr_tags)
assert len(name2crels) == len(all_essays)


# initial settings for other params
best_top_n, best_C, best_max_upd, best_max_parses, best_min_prob = (2, 0.0025, 2, 300, 0.0)  # min prob of 0 seems better

# In[93]:

xs_rerank = essay_to_crels_cv(cv_folds, models, top_n=best_top_n, search_mode_max_prob=False)
xs = get_features_from_probabilities(xs_rerank, name2crels, best_max_parses, min_feat_freq=1, min_prob=best_min_prob)

cv_folds_rerank = cross_validation(xs, 5)
cv_folds_mm = [min_max_normalize_feats(train, test) for (train, test) in cv_folds_rerank]

f1 = train_model_parallel(cv_folds=cv_folds_mm, C=best_C, pa_type=1, loss_type="ml", max_update_items=best_max_upd, set_cr_tags=set_cr_tags)
f1  # 0.7421167703055035

# training data comes from the test fold predictions from CV on the full training dataset
xs_train = []
for train, test in cv_folds_rerank:
    xs_train.extend(test)

xs_test_rerank = essay_to_crels_cv(test_folds, final_test_model, top_n=best_top_n, search_mode_max_prob=False)
xs_test = get_features_from_probabilities(xs_test_rerank, name2crels, best_max_parses, min_feat_freq=1,
                                          min_prob=best_min_prob)
# Normalize both using training data
xs_train_mm, xs_test_mm = min_max_normalize_feats(xs_train,xs_test)

# partition data
num_train = int(0.8 * len(xs_train_mm))
tmp_train_copy = list(xs_train_mm)
np.random.shuffle(tmp_train_copy)
tmp_train, tmp_test = tmp_train_copy[:num_train], tmp_train_copy[num_train:]

# Train With Early Stopping to Determine number of iterations
C = best_C
pa_type = 1
loss_type= "ml"
max_update_items = best_max_upd

mdl = CostSensitiveMIRA(C=C, pa_type=pa_type, loss_type=loss_type,
                        max_update_items=max_update_items, initial_weight=0.01)

best_mdl, test_acc_df_ml = train_model(mdl, xs_train=tmp_train, xs_test=tmp_test, name2essay=name2essay, set_cr_tags=set_cr_tags,
     max_epochs=20, early_stop_iters=5, train_instance_fn = train_cost_sensitive_instance, verbose=True)

