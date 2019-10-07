from Decorators import memoize_to_disk
from load_data import load_process_essays

from CrossValidation import cross_validation
from results_procesor import ResultsProcessor
from tag_frequency import get_tag_freq, regular_tag
from window_based_tagger_config import get_config
from nltk_featureextractionfunctions import *

from collections import defaultdict
from joblib import Parallel, delayed

from nltk.tag.crf import CRFTagger
from wordtagginghelper import merge_dictionaries
from nltk_datahelper import to_sentences, to_flattened_binary_tags_by_code, to_most_common_code_tagged_sentences, \
    tally_code_frequencies
from nltk_datahelper import to_label_powerset_tagged_sentences
from random import randint

import Settings
import logging, os

def train_classifer_on_fold(essays_TD, essays_VD, regular_tags, fold, code_freq, training_opt):

    # Start Training
    # Start Training
    print("Fold %i Training code" % fold)

    # For training
    td_sents = to_most_common_code_tagged_sentences(essays_TD, regular_tags, code_freq)
    vd_sents = to_most_common_code_tagged_sentences(essays_VD, regular_tags, code_freq)

    model_filename = models_folder + "/" + "%i_%s__%s" % (fold, "most_freq_code", str(randint(0, 9999999)))

    model = CRFTagger(feature_func=comp_feat_extactor, verbose=False, training_opt=training_opt)
    model.train(td_sents, model_filename)

    td_predictions = model.tag_sents(to_sentences(td_sents))
    vd_predictions = model.tag_sents(to_sentences(vd_sents))

    # for evaluation - binary tags
    # YS (ACTUAL)
    td_sents_pset = to_label_powerset_tagged_sentences(essays_TD, regular_tags)
    vd_sents_pset = to_label_powerset_tagged_sentences(essays_VD, regular_tags)

    wd_td_ys_bytag = to_flattened_binary_tags_by_code(td_sents_pset, regular_tags)
    wd_vd_ys_bytag = to_flattened_binary_tags_by_code(vd_sents_pset, regular_tags)

    # YS (PREDICTED)
    td_wd_predictions_by_code = to_flattened_binary_tags_by_code(td_predictions, regular_tags)
    vd_wd_predictions_by_code = to_flattened_binary_tags_by_code(vd_predictions, regular_tags)
    os.remove(model_filename)

    return wd_td_ys_bytag, wd_vd_ys_bytag, td_wd_predictions_by_code, vd_wd_predictions_by_code

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Load the Essays
# ---------------
# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS = True

MIN_FEAT_FREQ = 5  # 5 best so far
CV_FOLDS = 5

MIN_TAG_FREQ = 5
LOOK_BACK = 0  # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder                     = root_folder + "Training/"
test_folder                         = root_folder + "Test/"

models_folder = settings.data_directory + "CoralBleaching/models/CRF"


""" Load Configs """
train_config = get_config(training_folder)
train_config["window_size"] = 9
offset = (train_config["window_size"] - 1) / 2

test_config = dict(train_config.items())
test_config["folder"] = test_folder

""" Load Data """
train_tagged_essays = load_process_essays(**train_config)
test_tagged_essays = load_process_essays(**test_config)

logger.info("Essays loaded - Train: %i Test %i" % (len(train_tagged_essays), len(test_tagged_essays)))

# Create Corpus in CRF Format (list of list of tuples(word,tag))
# --------------------------------------------------------------

""" Define Tags """
tag_freq = get_tag_freq(train_tagged_essays)
regular_tags = list(set((tag for tag, freq in tag_freq.items() if freq >= 0 and tag[0].isdigit())))

""" FEATURE EXTRACTION """

unigram_stem_features = fact_extract_positional_word_features(offset, True)
trigram_stem_featues   = fact_extract_ngram_features(offset=offset, ngram_size=3, stem_words=True)
bigram_stem_featues   = fact_extract_ngram_features(offset=offset, ngram_size=2, stem_words=True)
unigram_bow_window_unstemmed = fact_extract_ngram_features(offset=offset, ngram_size=1, positional=False, stem_words=False)

extractors = [
    unigram_stem_features,
    bigram_stem_featues,
    trigram_stem_featues,
    unigram_bow_window_unstemmed,
    extract_brown_cluster,
    extract_dependency_relation
]

comp_feat_extactor = fact_composite_feature_extractor(extractors)

code_freq = tally_code_frequencies(train_tagged_essays)
folds = [(train_tagged_essays, test_tagged_essays)]

for feat_poss_state in [False]:
    for feat_poss_transitions in [True]:
        for c2 in [0.1]:
            cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
            cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

            training_opt = {"feature.possible_states" :     feat_poss_state,
                            "feature.possible_transitions": feat_poss_transitions,
                            "c2": c2
                            }

            training_opt_copy = dict([(k.replace(".", "_"),v) for k,v in training_opt.items()])

            results = Parallel(n_jobs=len(folds))(
                        delayed(train_classifer_on_fold)(essays_TD, essays_VD, regular_tags, fold, code_freq, training_opt)
                            for fold, (essays_TD, essays_VD) in enumerate(folds))

            for result in results:
                wd_td_ys_bytag, wd_vd_ys_bytag, td_wd_predictions_by_code, vd_wd_predictions_by_code = result

                merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
                merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
                merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
                merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

            logger.info("Training completed")

            """ Persist Results to Mongo DB """
            wd_algo = "CRF_MOST_COMMON_TAG"
            SUFFIX = "_CRF_MOST_COMMON_TAG_FIXED"
            CB_TAGGING_TD, CB_TAGGING_VD= "TEST_CB_TAGGING_TD" + SUFFIX, "TEST_CB_TAGGING_VD" + SUFFIX

            parameters = dict(train_config)
            parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
            parameters["min_feat_freq"] = MIN_FEAT_FREQ
            parameters.update(training_opt_copy)

            wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
            wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

            # This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
            print(processor.results_to_string(wd_td_objectid, CB_TAGGING_TD, wd_vd_objectid, CB_TAGGING_VD, "TAGGING"))
            logger.info("Results Processed")

