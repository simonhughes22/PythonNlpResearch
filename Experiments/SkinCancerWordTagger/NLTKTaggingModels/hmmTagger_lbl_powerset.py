from Decorators import memoize_to_disk
from load_data import load_process_essays

from CrossValidation import cross_validation
from results_procesor import ResultsProcessor
from tag_frequency import get_tag_freq, regular_tag
from window_based_tagger_config import get_config
from nltk_featureextractionfunctions import *

from collections import defaultdict
from joblib import Parallel, delayed

from nltk.tag.hmm import HiddenMarkovModelTrainer
from wordtagginghelper import merge_dictionaries
from nltk_datahelper import to_sentences, to_flattened_binary_tags_by_code
from nltk_datahelper import to_label_powerset_tagged_sentences

import Settings
import logging

def train_classifer_on_fold(essays_TD, essays_VD, regular_tags, fold):

    projection = lambda x: x
    if STEM:
        projection = stem

    # Start Training
    print("Fold %i Training code" % fold)

    # For training
    td_sents = to_label_powerset_tagged_sentences(essays_TD, regular_tags, projection=projection)
    vd_sents = to_label_powerset_tagged_sentences(essays_VD, regular_tags, projection=projection)

    trainer = HiddenMarkovModelTrainer()
    model = trainer.train_supervised(td_sents)

    td_predictions = model.tag_sents(to_sentences(td_sents))
    vd_predictions = model.tag_sents(to_sentences(vd_sents))

    # for evaluation - binary tags
    # YS (ACTUAL)
    wd_td_ys_bytag = to_flattened_binary_tags_by_code(td_sents, regular_tags)
    wd_vd_ys_bytag = to_flattened_binary_tags_by_code(vd_sents, regular_tags)

    # YS (PREDICTED)
    td_wd_predictions_by_code = to_flattened_binary_tags_by_code(td_predictions, regular_tags)
    vd_wd_predictions_by_code = to_flattened_binary_tags_by_code(vd_predictions, regular_tags)

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

STEM = True
# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()
root_folder = settings.data_directory + "SkinCancer/Thesis_Dataset/"
folder =                            root_folder + "Training/"
processed_essay_filename_prefix =   root_folder + "Pickled/essays_proc_pickled_"

config = get_config(folder)
print(config)

mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays(**config)
logger.info("Essays loaded")
len(tagged_essays)

# Create Corpus in CRF Format (list of list of tuples(word,tag))
# --------------------------------------------------------------

tag_freq = get_tag_freq(tagged_essays)
freq_tags = list(set((tag for tag, freq in tag_freq.items() if freq >= MIN_TAG_FREQ and regular_tag(tag))))
regular_tags = [t for t in freq_tags if t[0].isdigit()]

""" FEATURE EXTRACTION """
config["window_size"] = 11
offset = (config["window_size"] - 1) / 2

cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)
folds = cross_validation(tagged_essays, CV_FOLDS)

results = Parallel(n_jobs=CV_FOLDS)(
            delayed(train_classifer_on_fold)(essays_TD, essays_VD, regular_tags, fold)
                for fold, (essays_TD, essays_VD) in enumerate(folds))

for result in results:
    wd_td_ys_bytag, wd_vd_ys_bytag, td_wd_predictions_by_code, vd_wd_predictions_by_code = result

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

logger.info("Training completed")

""" Persist Results to Mongo DB """
wd_algo = "HMM_LBL_POWERSET"
SUFFIX = "_HMM_LBL_POWERSET"
SC_TAGGING_TD, SC_TAGGING_VD= "SC_TAGGING_TD" + SUFFIX, "SC_TAGGING_VD" + SUFFIX

parameters = dict(config)
parameters["min_feat_freq"] = MIN_FEAT_FREQ
if STEM:
    parameters["extractors"] = "stemmed_unigrams"
else:
    parameters["extractors"] = "unigrams"


wd_td_objectid = processor.persist_results(SC_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(SC_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

# This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
print processor.results_to_string(wd_td_objectid, SC_TAGGING_TD, wd_vd_objectid, SC_TAGGING_VD, "TAGGING")
logger.info("Results Processed")

