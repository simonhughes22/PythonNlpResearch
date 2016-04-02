from Decorators import memoize_to_disk
from GWCodes import GWConceptCodes
from TagTransformer import replace_periods
from sent_feats_for_stacking import *
from load_data import load_process_essays, extract_features

from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from predictions_to_file import predictions_to_file
from results_procesor import ResultsProcessor

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from window_based_tagger_config import get_config
from tag_frequency import get_tag_freq, regular_tag
# END Classifiers

import Settings
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized
processor = ResultsProcessor()

# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True

MIN_FEAT_FREQ       = 2        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags

settings = Settings.Settings()

root     =  settings.data_directory + "/GlobalWarming/BrattFiles/merged/"

""" INPUT - two serialized files, one for the pre-processed essays, the other for the features """

""" OUTPUT """
processed_essay_filename_prefix =  root + "Pickled/essays_proc_pickled_"
features_filename_prefix =         root + "Pickled/feats_pickled_"

out_predictions_file        = root + "Experiment/Output/predictions.txt"
out_predicted_margins_file  = root + "Experiment/Output/predicted_confidence.txt"
out_metrics_file            = root + "Experiment/Output/metrics.txt"
out_categories_file         = root + "Experiment/Output/categories.txt"

config = get_config(root)

""" FEATURE EXTRACTION """
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

extractors = [unigram_window_stemmed, biigram_window_stemmed]
feat_config = dict(config.items() + [("extractors", extractors)])

""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
# replace periods in tags so that we can store results in mongo
replace_periods(tagged_essays)
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk
mem_extract_features = memoize_to_disk(filename_prefix=features_filename_prefix)(extract_features)
essay_feats = mem_extract_features(tagged_essays, **feat_config)
logger.info("Features loaded")

""" DEFINE TAGS """

""" DEFINE TAGS """
gw_codes = GWConceptCodes()

tag_freq = get_tag_freq(tagged_essays)
freq_tags = list(set((tag for tag, freq in tag_freq.items()
                      if freq >= MIN_TAG_FREQ)))

valid_tags = list((t for t in freq_tags if gw_codes.is_valid_code(t)))

non_causal  = [t for t in valid_tags if "->" not in t]
only_causal = [t for t in valid_tags if "->" in t]

_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
regular_tags = list(set((t for t in flatten(lst_all_tags)
                         if "->" not in t and ":" not in t
                         and gw_codes.is_valid_code(t))))

CAUSE_TAGS = ["Causer", "Result", "explicit"]
CAUSAL_REL_TAGS = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]# + ["explicit"]

""" works best with all the pair-wise causal relation codes """
# Include all tags for the output
wd_train_tags = list(set(freq_tags + CAUSE_TAGS))
wd_test_tags  = list(set(freq_tags + CAUSE_TAGS))

# tags from tagging model used to train the stacked model
sent_input_feat_tags = list(set(freq_tags + CAUSE_TAGS))
# find interactions between these predicted tags from the word tagger to feed to the sentence tagger
sent_input_interaction_tags = list(set(non_causal + CAUSE_TAGS))
# tags to train (as output) for the sentence based classifier

#ONLY CAUSAL
#sent_output_train_test_tags = list(set(only_causal + CAUSE_TAGS + CAUSAL_REL_TAGS))

#CAUSAL + CONCEPT CODES
sent_output_train_test_tags = list(set(regular_tags + only_causal + CAUSE_TAGS + CAUSAL_REL_TAGS))
#sent_output_train_test_tags = list(set(only_causal))

assert set(CAUSE_TAGS).issubset(set(sent_input_feat_tags)), "To extract causal relations, we need Causer tags"
# tags to evaluate against

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """
fn_create_wd_cls = lambda: LogisticRegression() # C=1, dual = False seems optimal
#fn_create_wd_cls    = lambda : LinearSVC(C=1.0)

#fn_create_sent_cls  = lambda : LinearSVC(C=1.0)
fn_create_sent_cls  = lambda : LogisticRegression(dual=True) # C around 1.0 seems pretty optimal
# NOTE - GBT is stochastic in the SPLITS, and so you will get non-deterministic results
#fn_create_sent_cls  = lambda : GradientBoostingClassifier() #F1 = 0.5312 on numeric + 5b + casual codes for sentences

if type(fn_create_sent_cls()) == GradientBoostingClassifier:
    SPARSE_SENT_FEATS = False

f_output_file = open(out_predictions_file, "w+")
f_output_file.write("Essay|Sent Number|Processed Sentence|Concept Codes|Predictions\n")

# Gather metrics per fold
cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag = defaultdict(list), defaultdict(list)
cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag = defaultdict(list), defaultdict(list)

folds = cross_validation(essay_feats, CV_FOLDS)
#TODO Parallelize
for i,(essays_TD, essays_VD) in enumerate(folds):

    # TD and VD are lists of Essay objects. The sentences are lists
    # of featureextractortransformer.Word objects
    print "\nFold %s" % i
    print "Training Tagging Model"
    """ Data Partitioning and Training """
    td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
    vd_feats, vd_tags = flatten_to_wordlevel_feat_tags(essays_VD)

    feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)
    td_X, vd_X = feature_transformer.fit_transform(td_feats), feature_transformer.transform(vd_feats)
    wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_train_tags)

    """ TRAIN Tagger """
    tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, fn_create_wd_cls, wd_train_tags)

    """ TEST Tagger """
    td_wd_predictions_by_code = test_classifier_per_code(td_X, tag2word_classifier, wd_test_tags)
    vd_wd_predictions_by_code = test_classifier_per_code(vd_X, tag2word_classifier, wd_test_tags)

    print "\nTraining Sentence Model"
    """ SENTENCE LEVEL PREDICTIONS FROM STACKING """
    sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, essays_TD, td_X, wd_td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)
    sent_vd_xs, sent_vd_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, essays_VD, vd_X, wd_vd_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

    """ Train Stacked Classifier """
    tag2sent_classifier = train_classifier_per_code(sent_td_xs, sent_td_ys_bycode , fn_create_sent_cls, sent_output_train_test_tags)

    """ Test Stack Classifier """
    td_sent_predictions_by_code \
        = test_classifier_per_code(sent_td_xs, tag2sent_classifier, sent_output_train_test_tags )

    vd_sent_predictions_by_code \
        = test_classifier_per_code(sent_vd_xs, tag2sent_classifier, sent_output_train_test_tags )

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    merge_dictionaries(sent_td_ys_bycode, cv_sent_td_ys_by_tag)
    merge_dictionaries(sent_vd_ys_bycode, cv_sent_vd_ys_by_tag)
    merge_dictionaries(td_sent_predictions_by_code, cv_sent_td_predictions_by_tag)
    merge_dictionaries(vd_sent_predictions_by_code, cv_sent_vd_predictions_by_tag)

    predictions_to_file(f_output_file, sent_vd_ys_bycode, vd_sent_predictions_by_code, essays_VD, codes=sent_output_train_test_tags)

f_output_file.close()
# print results for each code
logger.info("Training completed")

""" Persist Results to Mongo DB """

wd_algo   = str(fn_create_wd_cls())
sent_algo = str(fn_create_sent_cls())

SUFFIX = "_CAUSE_EFFECT_LBLS"
GW_TAGGING_TD, GW_TAGGING_VD, GW_SENT_TD, GW_SENT_VD = "GW_TAGGING_TD" + SUFFIX, "GW_TAGGING_VD" + SUFFIX, "GW_SENT_TD" + SUFFIX, "GW_SENT_VD" + SUFFIX
parameters = dict(config)
parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
parameters["min_feat_freq"] = MIN_FEAT_FREQ


wd_td_objectid = processor.persist_results(GW_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(GW_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

sent_td_objectid = processor.persist_results(GW_SENT_TD, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, parameters, sent_algo, tagger_id=wd_td_objectid)
sent_vd_objectid = processor.persist_results(GW_SENT_VD, cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag, parameters, sent_algo, tagger_id=wd_vd_objectid)

# This outputs 0's for MEAN CONCEPT CODES as we aren't including those in the outputs

print processor.results_to_string(wd_td_objectid,   GW_TAGGING_TD,  wd_vd_objectid,     GW_TAGGING_VD,  "TAGGING")
print processor.results_to_string(sent_td_objectid, GW_SENT_TD,     sent_vd_objectid,   GW_SENT_VD,     "SENTENCE")
