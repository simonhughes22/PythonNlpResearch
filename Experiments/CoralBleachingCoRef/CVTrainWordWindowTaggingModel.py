from collections import Counter

from sklearn.linear_model import LogisticRegression
from CrossValidation import cross_validation
from Decorators import memoize_to_disk
from FindFiles import find_files
from Settings import Settings
from featureextractionfunctions import fact_extract_positional_word_features_stemmed, \
    fact_extract_ngram_features_stemmed, fact_extract_bow_ngram_features
from load_data import load_process_essays, extract_features
from results_procesor import ResultsProcessor, __MICRO_F1__
from window_based_tagger_config import get_config
from CoRefHelper import get_processed_essays
from TrainingHelper import train_tagger
from wordtagginghelper import *
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

processor = ResultsProcessor(dbname="metrics_coref")

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
partition = "Training" # Training | Test
target_folder = root_folder + partition + "/"
processed_essay_filename_prefix =  root_folder + "Pickled/essays_proc_pickled_"

config = get_config(target_folder)

# LOAD ESSAYS
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays(**config)
# map parsed essays to essay name

print("{0} essays loaded".format(len(tagged_essays)))

# LOAD COREF RESULTS
coref_root = root_folder + "CoReference/"
coref_folder = coref_root + partition
coref_files = find_files(coref_folder, ".*\.tagged")
print("{0} co-ref tagged files loaded".format(len(coref_files)))
assert len(coref_files) == len(tagged_essays)

config["window_size"] = 9
offset = int((config["window_size"] - 1) / 2)

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

    # extract_dependency_relation,
    # extract_brown_cluster
]

# For mongo
extractor_names = list(map(lambda fn: fn.func_name, extractors))
print("Extractors\n\t" + "\n\t".join(extractor_names))

def evaluate_tagger(dual, C, penalty, fit_intercept, multi_class, max_mention_len, max_reference_len, must_have_noun_phrase, tag_freq):
    hyper_opt_params = dict(locals())
    del hyper_opt_params["tag_freq"]

    # Gather metrics per fold
    cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
    cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

    """ This doesn't run in parallel ! """
    for fold, (essays_TD, essays_VD) in enumerate(folds):

        result = train_tagger(fold, essays_TD, essays_VD, wd_test_tags, wd_train_tags,
                              dual=dual, C=C, penalty=penalty, fit_intercept=fit_intercept,
                              multi_class=multi_class, tag_freq=tag_freq)

        td_wd_predictions_by_code, vd_wd_predictions_by_code, wd_td_ys_bytag, wd_vd_ys_bytag = result
        merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
        merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
        merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
        merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    # print results for each code
    """ Persist Results to Mongo DB """

    SUFFIX = "_WINDOW_CLASSIFIER_MOST_COMMON_TAG_MULTICLASS"
    CB_TAGGING_TD, CB_TAGGING_VD = "CB_TAGGING_TD" + SUFFIX, "CB_TAGGING_VD" + SUFFIX
    parameters = dict(config)
    parameters["extractors"] = list(map(lambda fn: fn.func_name, extractors))
    parameters["min_feat_freq"] = MIN_FEAT_FREQ
    parameters.update(hyper_opt_params)

    wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
    wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

    # This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
    avg_f1 = float(processor.get_metric(CB_TAGGING_VD, wd_vd_objectid, __MICRO_F1__)["f1_score"])
    return avg_f1

feat_config = dict(list(config.items()) + [("extractors", extractors)])

# for max_reference_len in [1, 2, 3, 5, 10, 100]:
#     for max_mention_len in [1, 2, 3, 5, 10, 100]:
for max_reference_len in [0]:
    for max_mention_len in [0]:
        # for must_not_have_noun_phrase in [True, False]: # Don't replace if there is one or more real noun phrases in the reference
        for must_not_have_noun_phrase in [True]: # Don't replace if there is one or more real noun phrases in the reference
            updated_essays = get_processed_essays(tagged_essays, coref_files,
                                                  max_mention_len=max_mention_len, max_reference_len=max_reference_len,
                                                  must_not_have_noun_phrase=must_not_have_noun_phrase)
            """ LOAD DATA """
            assert len(updated_essays) == len(tagged_essays), "Must be same number of essays after processing"
            print(len(updated_essays), "updated essays")

            # most params below exist ONLY for the purposes of the hashing to and from disk
            train_essay_feats = extract_features(updated_essays, **feat_config)
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

            folds = cross_validation(train_essay_feats, CV_FOLDS)

            """ CLASSIFIERS """
            fn_create_wd_cls   = lambda: LogisticRegression(n_jobs=-1) # C=1, dual = False seems optimal
            wd_algo   = str(fn_create_wd_cls())

            dual = True
            fit_intercept = True
            penalty = "l2"
            C = 0.5
            multi_class  = 'ovr'

            avg_f1 = evaluate_tagger(dual=dual, C=C, penalty=penalty,
                                     fit_intercept=fit_intercept, multi_class=multi_class,
                                     max_reference_len=max_reference_len, max_mention_len=max_reference_len,
                                     must_have_noun_phrase=must_not_have_noun_phrase, tag_freq=tag_freq)

            logger.info("AVG F1: {f1:.6f}, Max Mention: {mention} Max Ref: {reference} Has NP: {has_np}".format(
                f1=avg_f1, mention=max_mention_len, reference=max_reference_len, has_np=must_not_have_noun_phrase
            ))