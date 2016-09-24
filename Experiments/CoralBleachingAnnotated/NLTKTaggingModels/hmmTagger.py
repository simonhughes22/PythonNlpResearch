from Decorators import memoize_to_disk
from load_data import load_process_essays

from CrossValidation import cross_validation
from results_procesor import ResultsProcessor
from tag_frequency import get_tag_freq, regular_tag
from window_based_tagger_config import get_config
from nltk_featureextractionfunctions import stem

from collections import defaultdict

from nltk.tag.hmm import HiddenMarkovModelTrainer
from wordtagginghelper import merge_dictionaries
from nltk_datahelper import to_sentences, to_flattened_binary_tags, to_tagged_sentences_by_code

import Settings
import logging, os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Load the Essays
# ---------------
# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

CV_FOLDS = 5

MIN_TAG_FREQ = 5
STEM = True
# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()
folder = settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
out_metrics_file = settings.data_directory + "CoralBleaching/Results/metrics.txt"

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
non_causal  = [t for t in freq_tags if "->" not in t]
only_causal = [t for t in freq_tags if "->" in t]
regular_tags = [t for t in freq_tags if t[0].isdigit()]

""" FEATURE EXTRACTION """
config["window_size"] = 1
offset = (config["window_size"] - 1) / 2

fold_models = []
cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

folds = cross_validation(tagged_essays, CV_FOLDS)

projection = lambda x : x
if STEM:
    projection = stem

#TODO Parallelize (if so, make sure model files are unique)
for fold, (essays_TD, essays_VD) in enumerate(folds):
    td_sents_by_code = to_tagged_sentences_by_code(essays_TD, regular_tags, projection=projection)
    vd_sents_by_code = to_tagged_sentences_by_code(essays_VD, regular_tags, projection=projection)

    code2model = dict()
    fold_models.append(code2model)

    wd_td_ys_bytag = dict()
    wd_vd_ys_bytag = dict()
    td_wd_predictions_by_code = dict()
    vd_wd_predictions_by_code = dict()

    for code in sorted(regular_tags):
        print("Fold %i Training code: %s" % (fold, code))
        td, vd = td_sents_by_code[code], vd_sents_by_code[code]

        trainer = HiddenMarkovModelTrainer()
        model = trainer.train_supervised(td)
        code2model[code] = model

        wd_td_ys_bytag[code] = to_flattened_binary_tags(td)
        wd_vd_ys_bytag[code] = to_flattened_binary_tags(vd)

        td_predictions = model.tag_sents(to_sentences(td))
        vd_predictions = model.tag_sents(to_sentences(vd))

        td_wd_predictions_by_code[code] = to_flattened_binary_tags(td_predictions)
        vd_wd_predictions_by_code[code] = to_flattened_binary_tags(vd_predictions)

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

logger.info("Training completed")

""" Persist Results to Mongo DB """

wd_algo = "HMM"
SUFFIX = "_HMM"
CB_TAGGING_TD, CB_TAGGING_VD= "CB_TAGGING_TD" + SUFFIX, "CB_TAGGING_VD" + SUFFIX
parameters = dict(config)
if STEM:
    parameters["extractors"] = "stemmed_unigrams"
else:
    parameters["extractors"] = "unigrams"

wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters,
                                           wd_algo)
wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters,
                                           wd_algo)

# This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs

print processor.results_to_string(wd_td_objectid, CB_TAGGING_TD, wd_vd_objectid, CB_TAGGING_VD, "TAGGING")
logger.info("Results Processed")

