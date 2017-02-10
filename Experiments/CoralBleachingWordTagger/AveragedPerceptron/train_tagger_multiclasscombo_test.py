# coding=utf-8
import logging
from collections import Counter

from joblib import Parallel
from joblib import delayed

import Settings
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from Rpfa import micro_rpfa
from featureextractionfunctions import *
from load_data import load_process_essays, extract_features
from perceptron_tagger_multiclass_combo import PerceptronTaggerMultiClassCombo
from perceptron_tagger_multiclass_combo_new import PerceptronTaggerLabelPowerset
from results_procesor import ResultsProcessor, __MICRO_F1__
from window_based_tagger_config import get_config
from wordtagginghelper import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 1

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags

# Combo tags
#NOTE: this essentially forces it to ignore lbl powersets
#TAG_FREQ_THRESHOLD   = 5

# end not hashed

# construct unique key using settings for pickling

#TODO Set training data and test data as a single fold
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
logger.info("Essays loaded - Train: %i Test %i" % (len(train_tagged_essays), len(test_tagged_essays)))

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

""" CLASSIFIERS """
# folds = cross_validation(train_essay_feats, CV_FOLDS)
folds = [(train_essay_feats, test_essay_feats)]

""" Prepare the folds (pre-process to avoid unnecessary computation) """
# Build this up once
cv_wd_td_ys_by_tag, cv_wd_vd_ys_by_tag = defaultdict(list), defaultdict(list)
# Store the random pickled file names for use in training
k_fold_2data = {}
for kfold, (essays_TD, essays_VD) in enumerate(folds):
    """ Compute the target labels (ys) """
    # Just get the tags (ys)
    _, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    _, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)
    wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)

    """ Transform TD to only have most frequent tags """
    essays_TD_most_freq = essaysfeats_to_most_common_tags(essays_TD, tag_freq=tag_freq)

    k_fold_2data[kfold] = (essays_TD, essays_VD, essays_TD_most_freq, wd_td_ys_bytag, wd_vd_ys_bytag)

def evaluate_tagger_on_fold(kfold, wd_train_tags, tag_history, tag_plus_word, tag_ngram, split=0.2):

    # logger.info("Loading data for fold %i" % kfold)
    k_fold_data = k_fold_2data[kfold]
    essays_TD, essays_VD, essays_TD_most_freq, wd_td_ys_bytag, wd_vd_ys_bytag = k_fold_data

    """ TRAINING """
    tagger = PerceptronTaggerLabelPowerset(wd_train_tags,
                                           combo_freq_threshold=1,
                                           tag_history=tag_history,
                                           tag_plus_word=tag_plus_word,
                                           tag_ngram_size=tag_ngram)

    # Split into train and test set
    np_essays = np.asarray(essays_TD_most_freq)
    ixs = np.arange(len(essays_TD_most_freq))
    np.random.shuffle(ixs)
    split_size = int(split * len(essays_TD_most_freq))

    test, train = np_essays[ixs[:split_size]], np_essays[ixs[split_size:]]
    _, test_tags = flatten_to_wordlevel_feat_tags(test)
    class2ys = get_wordlevel_ys_by_code(test_tags, wd_train_tags)

    optimal_num_iterations = -1
    last_f1 = -1
    """ EARLY STOPPING USING TEST SET """
    for i in range(30):
        tagger.train(train, nr_iter=1, verbose=False, average_weights=False)
        wts_copy = dict(tagger.model.weights.items())
        tagger.model.average_weights()

        class2predictions = tagger.predict(test)
        #Compute F1 score, stop early if worse than previous
        class2metrics = ResultsProcessor.compute_metrics(class2ys, class2predictions)
        micro_metrics = micro_rpfa(class2metrics.values())
        current_f1 = micro_metrics.f1_score
        if current_f1 <= last_f1:
            optimal_num_iterations = i # i.e. this number minus 1, but 0 based
            break
        # Reset weights (as we are averaging weights)
        tagger.model.weights = wts_copy
        last_f1 = current_f1

    # print("fold %i - Optimal F1 obtained at iteration %i " % (kfold, optimal_num_iterations))
    """ Re-train model using stopping criterion on full training set """
    final_tagger = PerceptronTaggerLabelPowerset(wd_train_tags,
                                           combo_freq_threshold=1,
                                           tag_history=tag_history,
                                           tag_plus_word=tag_plus_word,
                                           tag_ngram_size=tag_ngram)

    final_tagger.train(essays_TD_most_freq, nr_iter=optimal_num_iterations, verbose=False)

    """ PREDICT """
    td_wd_predictions_by_code = final_tagger.predict(essays_TD)
    vd_wd_predictions_by_code = final_tagger.predict(essays_VD)

    # logger.info("Fold %i finished" % kfold)
    """ Aggregate results """
    return kfold, td_wd_predictions_by_code, vd_wd_predictions_by_code, optimal_num_iterations

def evaluate_tagger(wd_train_tags, tag_history, tag_plus_word, tag_ngram):

    """ Run K Fold CV in parallel """
    print("\nNew Run - Tag History: %i\tTag + Wd: %i\tTag Ngram: %i" % (tag_history, tag_plus_word, tag_ngram))

    results = Parallel(n_jobs=(len(folds)))(
         delayed(evaluate_tagger_on_fold)(kfold, wd_train_tags, tag_history, tag_plus_word, tag_ngram)
            for kfold in range(len(folds)))

    # Merge results of parallel processing
    cv_wd_td_predictions_by_tag, cv_wd_vd_predictions_by_tag    = defaultdict(list), defaultdict(list)
    # important to sort by k value

    optimal_traning_iterations = []
    for kf, td_wd_predictions_by_code, vd_wd_predictions_by_code, opt_iter in sorted(results, key = lambda (k, td, vd, iter): k):
        optimal_traning_iterations.append(opt_iter)
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)
        pass

    suffix = "_AVG_PERCEPTRON_MOST_COMMON_TAG"
    CB_TAGGING_TD, CB_TAGGING_VD = "TEST_CB_TAGGING_TD" + suffix, "TEST_CB_TAGGING_VD" + suffix
    parameters = dict(train_config)
    #parameters["prev_tag_sharing"] = True  # don't include tags from other binary models
    """ False: 0.737 - 30 iterations """
    parameters["tag_history"]    = tag_history
    parameters["tag_plus_word"]  = tag_plus_word
    parameters["tag_ngram_size"] = tag_ngram

    # store optimal number of iterations from early stopping. Not really parameters
    parameters["early_stopping_training_iterations"] = optimal_traning_iterations
    #parameters["combo_freq_threshold"] = TAG_FREQ_THRESHOLD

    parameters["extractors"] = extractor_names
    wd_algo = "AveragedPerceptronMultiClass_TagHistoryFixed"

    _              = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

    avg_f1 = float(processor.get_metric(CB_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1

best_f1 = 0
for tag_history in [1]:
    for tag_plus_word in [5]:
        for tag_ngram in [1]:

            new_f1 = evaluate_tagger(wd_train_tags=wd_train_tags,
                                     tag_history=tag_history,
                                     tag_plus_word=tag_plus_word,
                                     tag_ngram=tag_ngram)
            if new_f1 > best_f1:
                best_f1 = new_f1
                print(("!" * 8) + " NEW BEST MICRO F1 " + ("!" * 8))
            print(" Micro F1 %f - Tag History: %i\tTag + Wd: %i\tTag Ngram: %i" % (new_f1, tag_history, tag_plus_word, tag_ngram))
