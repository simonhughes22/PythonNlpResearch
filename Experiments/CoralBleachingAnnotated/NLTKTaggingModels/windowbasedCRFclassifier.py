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
from nltk_datahelper import to_sentences, to_flattened_binary_tags, to_tagged_sentences_by_code
from random import randint

import Settings
import logging, os

def train_classifer_on_fold(essays_TD, essays_VD, regular_tags, fold):
    td_sents_by_code = to_tagged_sentences_by_code(essays_TD, regular_tags)
    vd_sents_by_code = to_tagged_sentences_by_code(essays_VD, regular_tags)

    wd_td_ys_bytag = dict()
    wd_vd_ys_bytag = dict()
    td_wd_predictions_by_code = dict()
    vd_wd_predictions_by_code = dict()

    for code in sorted(regular_tags):
        print("Fold %i Training code: %s" % (fold, code))
        td, vd = td_sents_by_code[code], vd_sents_by_code[code]

        model_filename = models_folder + "/" + "%i_%s__%s" % (fold, code, str(randint(0, 9999999)))

        # documentation: http://www.chokkan.org/software/crfsuite/manual.html
        model = CRFTagger(feature_func=comp_feat_extactor, verbose=False)
        model.train(td, model_filename)

        wd_td_ys_bytag[code] = to_flattened_binary_tags(td)
        wd_vd_ys_bytag[code] = to_flattened_binary_tags(vd)

        td_predictions = model.tag_sents(to_sentences(td))
        vd_predictions = model.tag_sents(to_sentences(vd))
        # Delete model file now predictions obtained
        # Note, we are randomizing name above, so we need to clean up here
        os.remove(model_filename)

        td_wd_predictions_by_code[code] = to_flattened_binary_tags(td_predictions)
        vd_wd_predictions_by_code[code] = to_flattened_binary_tags(vd_predictions)
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
folder = settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
processed_essay_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
features_filename_prefix = settings.data_directory + "CoralBleaching/BrattData/Pickled/feats_pickled_"
models_folder = settings.data_directory + "CoralBleaching/models/CRF"
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
config["window_size"] = 11
offset = (config["window_size"] - 1) / 2

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
wd_algo = "CRF"
SUFFIX = "_CRF"
CB_TAGGING_TD, CB_TAGGING_VD= "CB_TAGGING_TD" + SUFFIX, "CB_TAGGING_VD" + SUFFIX

parameters = dict(config)
parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
parameters["min_feat_freq"] = MIN_FEAT_FREQ

wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

# This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs
print processor.results_to_string(wd_td_objectid, CB_TAGGING_TD, wd_vd_objectid, CB_TAGGING_VD, "TAGGING")
logger.info("Results Processed")

