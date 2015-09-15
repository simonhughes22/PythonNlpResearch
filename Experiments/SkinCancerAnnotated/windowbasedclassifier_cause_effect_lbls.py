from Decorators import memoize_to_disk
from sent_feats_for_stacking import *
from load_data import load_process_essays, extract_features

from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
from DictionaryHelper import tally_items
from predictions_to_file import predictions_to_file
from predictions_to_console import predictions_to_console
from results_procesor import ResultsProcessor
from argument_hasher import argument_hasher
# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.lda import LDA

from window_based_tagger_config import get_config
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

MIN_FEAT_FREQ       = 10        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 10
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder  =                           settings.data_directory + "SkinCancer/EBA1415_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "SkinCancer/Pickled/essays_proc_pickled_"
features_filename_prefix =          settings.data_directory + "SkinCancer/Pickled/feats_pickled_"

out_metrics_file     =              settings.data_directory + "SkinCancer/Results/metrics.txt"
out_predictions_file =              settings.data_directory + "SkinCancer/Results/predictions.txt"

config = get_config(folder)

""" FEATURE EXTRACTION """
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

#pos_tag_window = fact_extract_positional_POS_features(offset)
#pos_tag_plus_wd_window = fact_extract_positional_POS_features_plus_word(offset)
#head_wd_window = fact_extract_positional_head_word_features(offset)
#pos_dep_vecs = fact_extract_positional_dependency_vectors(offset)

extractors = [unigram_window_stemmed, biigram_window_stemmed]
feat_config = dict(config.items() + [("extractors", extractors)])

""" LOAD DATA """
mem_process_essays = memoize_to_disk(filename_prefix=processed_essay_filename_prefix)(load_process_essays)
tagged_essays = mem_process_essays( **config )
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk
mem_extract_features = memoize_to_disk(filename_prefix=features_filename_prefix)(extract_features)
essay_feats = mem_extract_features(tagged_essays, **feat_config)
logger.info("Features loaded")

""" DEFINE TAGS """
tag_freq = defaultdict(int)
for essay in tagged_essays:
    for sentence in essay.sentences:
        un_tags = set()
        for word, tags in sentence:
            for tag in tags:
                if "5b" in tag:
                    continue
                if (tag[-1].isdigit() or tag in {"Causer", "explicit", "Result"} \
                        or tag.startswith("Causer") or tag.startswith("Result") or tag.startswith("explicit") or "->" in tag)\
                        and not ("Anaphor" in tag or "rhetorical" in tag or "other" in tag):
                #if not ("Anaphor" in tag or "rhetorical" in tag or "other" in tag):
                    un_tags.add(tag)
        for tag in un_tags:
            tag_freq[tag] += 1

all_tags = list(tag_freq.keys())
freq_tags = list(set((tag for tag, freq in tag_freq.items() if freq >= MIN_TAG_FREQ)))
non_causal  = [t for t in freq_tags if "->" not in t]
only_causal = [t for t in freq_tags if "->" in t]

_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
regular_tags = list(set((t for t in flatten(lst_all_tags) if t[0].isdigit())))

CAUSE_TAGS = ["Causer", "Result", "explicit"]
CAUSAL_REL_TAGS = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]# + ["explicit"]

""" works best with all the pair-wise causal relation codes """
# Include all tags for the output
wd_train_tags = list(set(all_tags + CAUSE_TAGS))
wd_test_tags  = list(set(all_tags + CAUSE_TAGS))
#wd_train_tags = list(set(freq_tags + CAUSE_TAGS))
#wd_test_tags  = list(set(freq_tags + CAUSE_TAGS))

# tags from tagging model used to train the stacked model
sent_input_feat_tags = list(set(freq_tags + CAUSE_TAGS))
# find interactions between these predicted tags from the word tagger to feed to the sentence tagger
sent_input_interaction_tags = list(set(non_causal + CAUSE_TAGS))
# tags to train (as output) for the sentence based classifier
#sent_output_train_test_tags = list(set(regular_tags + only_causal + CAUSE_TAGS + CAUSAL_REL_TAGS))
#sent_output_train_test_tags = list(set(only_causal + CAUSE_TAGS + CAUSAL_REL_TAGS))
sent_output_train_test_tags = list(set(all_tags + CAUSE_TAGS + CAUSAL_REL_TAGS))

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
SC_TAGGING_TD, SC_TAGGING_VD, SC_SENT_TD, SC_SENT_VD = "SC_TAGGING_TD" + SUFFIX, "SC_TAGGING_VD" + SUFFIX, "SC_SENT_TD" + SUFFIX, "SC_SENT_VD" + SUFFIX
parameters = dict(config)
parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
parameters["min_feat_freq"] = MIN_FEAT_FREQ


wd_td_objectid = processor.persist_results(SC_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(SC_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

sent_td_objectid = processor.persist_results(SC_SENT_TD, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, parameters, sent_algo, tagger_id=wd_td_objectid)
sent_vd_objectid = processor.persist_results(SC_SENT_VD, cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag, parameters, sent_algo, tagger_id=wd_vd_objectid)

print processor.results_to_string(wd_td_objectid,   SC_TAGGING_TD,  wd_vd_objectid,     SC_TAGGING_VD,  "TAGGING")
print processor.results_to_string(sent_td_objectid, SC_SENT_TD,     sent_vd_objectid,   SC_SENT_VD,     "SENTENCE")
logger.info("Results Processed")
"""
# PLAN
#   WORD LEVEL FEATURE EXTRACTION - use functions specific to the individual word, but that can look around at the
#       previous and next words and sentences if needed. This can handle every scenario where I want to leverage features
#       across sentences and at the essay level.
#   MEMOIZE SENTENCE LEVEL FEATS (e.g. deps) -  Will need memoizing when extracting dependency parse features per sentence (as called once for every word in sentence)
#   WORD \ SENTENCE PARTITIONING FOR WORD AND SENTENCE LEVEL TAGGING
#       Need a class that can transform the feature dictionaries (from essay structure form) into training and test data
#       for word tagging and also for sentence classifying. Suggest do k fold cross validation at the essay level.
#   LOAD RESULTS INTO A DB
#TODO Feed into the sentence classifier the number of words tagged with each category, the proportion of words (to control for sentence length variations) and
    also the number of contiguous segments of each in case some codes occur more than once (in separate segments - probably with > 1 word gaps in between)
#TODO Switch to micro and macro-average F1 scores as described in http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.8244&rep=rep1&type=pdf, page 6.
#TODO Look at using the StructureLINK relations (related to anaphora and other devices, see ReadMe.txt in the CoralBleaching folder).
#TODO Include dependency parse features
#TODO Parallelize the cross fold validation
>>> These results compute the mean across all regular tags (tagging model) and regular tags plus causal (sentence)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
TAGGING
Training   Performance
Weighted:Recall: 0.8021, Precision: 0.9443, F1: 0.8617, Accuracy: 0.9933, Codes:  4175
Mean    :Recall: 0.6790, Precision: 0.9194, F1: 0.7594, Accuracy: 0.9965, Codes:    13
Validation Performance
Weighted:Recall: 0.6375, Precision: 0.8561, F1: 0.7109, Accuracy: 0.9862, Codes:  5219
Mean    :Recall: 0.5141, Precision: 0.7584, F1: 0.5745, Accuracy: 0.9931, Codes:    65
GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=3, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=2, n_estimators=100,
              random_state=None, subsample=1.0, verbose=0,
              warm_start=False)
SENTENCE
Training   Performance
Weighted:Recall: 0.9924, Precision: 0.9875, F1: 0.9899, Accuracy: 0.9966, Codes:  2136
Mean    :Recall: 0.9947, Precision: 0.9858, F1: 0.9900, Accuracy: 0.9984, Codes:    16
Validation Performance
Weighted:Recall: 0.7381, Precision: 0.8234, F1: 0.7679, Accuracy: 0.9359, Codes:  2670
Mean    :Recall: 0.6790, Precision: 0.7513, F1: 0.6872, Accuracy: 0.9662, Codes:    80
****************************************************************************************
LR as the tagger (same tagging results)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
SENTENCE
Training   Performance
Weighted:Recall: 0.9575, Precision: 0.9520, F1: 0.9533, Accuracy: 0.9823, Codes:  2136
Mean    :Recall: 0.9788, Precision: 0.9660, F1: 0.9715, Accuracy: 0.9924, Codes:    16
Validation Performance
Weighted:Recall: 0.7399, Precision: 0.8122, F1: 0.7662, Accuracy: 0.9327, Codes:  2670
Mean    :Recall: 0.6717, Precision: 0.7553, F1: 0.6885, Accuracy: 0.9650, Codes:    80
****************************************************************************************
BEST !!!!!
LR as the tagger (same tagging results)
LR as the sentence classifier
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
SENTENCE
Training   Performance
Weighted:Recall: 0.9560, Precision: 0.9586, F1: 0.9572, Accuracy: 0.9845, Codes:  2136
Mean    :Recall: 0.9735, Precision: 0.9686, F1: 0.9708, Accuracy: 0.9932, Codes:    16
Validation Performance
Weighted:Recall: 0.7383, Precision: 0.8319, F1: 0.7749, Accuracy: 0.9365, Codes:  2670
Mean    :Recall: 0.6726, Precision: 0.7843, F1: 0.7033, Accuracy: 0.9673, Codes:    80
LR + LR
# DEPENDENCIES:
# head only
Validation Performance
Weighted:Recall: 0.7400, Precision: 0.8369, F1: 0.7776, Accuracy: 0.9370, Codes:  2670
Mean    :Recall: 0.6743, Precision: 0.7897, F1: 0.7064, Accuracy: 0.9677, Codes:    80
# head word only
***Validation Performance
Weighted:Recall: 0.7433, Precision: 0.8356, F1: 0.7787, Accuracy: 0.9383, Codes:  2670
Mean    :Recall: 0.6688, Precision: 0.7854, F1: 0.6995, Accuracy: 0.9679, Codes:    80
# head + target
Validation Performance
Weighted:Recall: 0.7382, Precision: 0.8313, F1: 0.7747, Accuracy: 0.9364, Codes:  2670
Mean    :Recall: 0.6667, Precision: 0.7863, F1: 0.7015, Accuracy: 0.9672, Codes:    80
# head only + child only
Validation Performance
Weighted:Recall: 0.7324, Precision: 0.8304, F1: 0.7703, Accuracy: 0.9358, Codes:  2670
Mean    :Recall: 0.6632, Precision: 0.7859, F1: 0.6962, Accuracy: 0.9668, Codes:    80
# child only
Validation Performance
Weighted:Recall: 0.7325, Precision: 0.8306, F1: 0.7714, Accuracy: 0.9359, Codes:  2670
Mean    :Recall: 0.6658, Precision: 0.7843, F1: 0.6983, Accuracy: 0.9669, Codes:    80
#child words only
Validation Performance
Weighted:Recall: 0.7316, Precision: 0.8343, F1: 0.7713, Accuracy: 0.9364, Codes:  2670
Mean    :Recall: 0.6624, Precision: 0.7859, F1: 0.6959, Accuracy: 0.9670, Codes:    80
# children + target
Validation Performance
Weighted:Recall: 0.7383, Precision: 0.8307, F1: 0.7744, Accuracy: 0.9362, Codes:  2670
Mean    :Recall: 0.6680, Precision: 0.7847, F1: 0.7012, Accuracy: 0.9672, Codes:    80
# head -> child
Validation Performance
Weighted:Recall: 0.7381, Precision: 0.8308, F1: 0.7745, Accuracy: 0.9356, Codes:  2670
Mean    :Recall: 0.6736, Precision: 0.7841, F1: 0.7031, Accuracy: 0.9670, Codes:    80
!!! FORGOT TO TURN OFF STEMMING !!!
@@@ 0.7591 @@@@
Regular NO STEM
Validation Performance
Weighted:Recall: 0.7198, Precision: 0.8226, F1: 0.7591, Accuracy: 0.9325, Codes:  2670
Mean    :Recall: 0.6534, Precision: 0.7761, F1: 0.6860, Accuracy: 0.9652, Codes:    80
#POSITIONAL HEAD
Validation Performance
Weighted:Recall: 0.7071, Precision: 0.8208, F1: 0.7517, Accuracy: 0.9309, Codes:  2670
Mean    :Recall: 0.6204, Precision: 0.7653, F1: 0.6599, Accuracy: 0.9643, Codes:    80
# DEP HEAD WORD ONLY
Validation Performance
Weighted:Recall: 0.7184, Precision: 0.8205, F1: 0.7579, Accuracy: 0.9324, Codes:  2670
Mean    :Recall: 0.6496, Precision: 0.7714, F1: 0.6825, Accuracy: 0.9649, Codes:    80
# NLTK POS TAGS
Validation Performance
Weighted:Recall: 0.7145, Precision: 0.8198, F1: 0.7538, Accuracy: 0.9316, Codes:  2670
Mean    :Recall: 0.6430, Precision: 0.7808, F1: 0.6789, Accuracy: 0.9647, Codes:    80
#NLT TAG ONLY (unary tag)
**** Validation Performance
Weighted:Recall: 0.7196, Precision: 0.8252, F1: 0.7596, Accuracy: 0.9330, Codes:  2670
Mean    :Recall: 0.6452, Precision: 0.7770, F1: 0.6808, Accuracy: 0.9654, Codes:    80
#NLTK POS + WORD
*****Validation Performance
Weighted:Recall: 0.7262, Precision: 0.8244, F1: 0.7636, Accuracy: 0.9338, Codes:  2670
Mean    :Recall: 0.6611, Precision: 0.7780, F1: 0.6913, Accuracy: 0.9658, Codes:    80
SWITCH TO STEMMING DURING FEAT EXTRACTION FOR THE MAIN INPUTS
    - Leaving sentence untouched for dep parser
STEMMED -
LR + LR + main feats only
Training   Performance
Weighted:Recall: 0.9563, Precision: 0.9597, F1: 0.9579, Accuracy: 0.9846, Codes:  2136
Mean    :Recall: 0.9745, Precision: 0.9704, F1: 0.9721, Accuracy: 0.9933, Codes:    16
Validation Performance
Weighted:Recall: 0.7378, Precision: 0.8306, F1: 0.7742, Accuracy: 0.9361, Codes:  2670
Mean    :Recall: 0.6742, Precision: 0.7848, F1: 0.7058, Accuracy: 0.9672, Codes:    80
add dual regularization to second LR
Validation Performance
Weighted:Recall: 0.7378, Precision: 0.8311, F1: 0.7744, Accuracy: 0.9362, Codes:  2670
Mean    :Recall: 0.6742, Precision: 0.7851, F1: 0.7059, Accuracy: 0.9672, Codes:    80
#NLTK POS + WD (worse with stemming on than stemming alone)
Validation Performance
Weighted:Recall: 0.7389, Precision: 0.8286, F1: 0.7737, Accuracy: 0.9353, Codes:  2670
Mean    :Recall: 0.6770, Precision: 0.7804, F1: 0.7063, Accuracy: 0.9670, Codes:    80
#NLTK Positional POS + WD
Validation Performance
Weighted:Recall: 0.7339, Precision: 0.8291, F1: 0.7696, Accuracy: 0.9345, Codes:  2670
Mean    :Recall: 0.6716, Precision: 0.7869, F1: 0.7021, Accuracy: 0.9665, Codes:    80
#Switch on dual Regularization on sentence classifier - slight performance boost
#Brown Cluster Only
Training   Performance
Weighted:Recall: 0.9593, Precision: 0.9621, F1: 0.9606, Accuracy: 0.9856, Codes:  2136
Mean    :Recall: 0.9766, Precision: 0.9712, F1: 0.9737, Accuracy: 0.9937, Codes:    16
Validation Performance
Weighted:Recall: 0.7365, Precision: 0.8238, F1: 0.7703, Accuracy: 0.9344, Codes:  2670
Mean    :Recall: 0.6712, Precision: 0.7806, F1: 0.7008, Accuracy: 0.9665, Codes:    80
#Brow Cluster + word
Validation Performance
Weighted:Recall: 0.7394, Precision: 0.8287, F1: 0.7739, Accuracy: 0.9355, Codes:  2670
Mean    :Recall: 0.6770, Precision: 0.7804, F1: 0.7063, Accuracy: 0.9670, Codes:    80
"""