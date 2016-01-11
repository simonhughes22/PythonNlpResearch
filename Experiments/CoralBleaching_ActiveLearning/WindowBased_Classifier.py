# coding=utf-8

# construct unique key using settings for pickling
import Settings

""" PETER - CHANGE THESE FILE PATHS """
root        = "/Users/simon.hughes/Google Drive/PhD/Data/ActiveLearning/"
f_training_essays = root + "training_essays.txt"
f_test_essays     = root + "test_essays.txt"

""" INPUT - two serialized files, one for the pre-processed essays, the other for the features """
serialized_features = root + "tmp_essay_feats.pl"
serialized_essays   = root + "tmp_essays.pl"

""" OUTPUT """
out_predictions_file = root + "output/predictions.txt"

""" END SETTINGS """

import cPickle as pickle
from sent_feats_for_stacking import *

from featurevectorizer import FeatureVectorizer

from wordtagginghelper import *
from IterableFP import flatten

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# END Classifiers
from tag_frequency import get_tag_freq, regular_tag
from predictions_to_file import predictions_to_file

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True

MIN_FEAT_FREQ       = 5        # 5 best so far

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags

""" LOAD DATA """
with open(serialized_essays, "r+") as f:
    tagged_essays = pickle.load(f)
logger.info("Essays loaded")

with open(serialized_features, "r+") as f:
    essay_feats = pickle.load(f)
logger.info("Features loaded")

with open(f_training_essays) as f:
    set_training_essays = set(map(lambda s: s.strip(), f.readlines()))

with open(f_test_essays) as f:
    set_test_essays = set(map(lambda s: s.strip(), f.readlines()))

""" Split essays according to lists """
train_essays, train_essay_feats = [], []
test_essays,  test_essay_feats  = [], []

for i, essay in enumerate(tagged_essays):
    feats = essay_feats[i]
    if essay.name in set_training_essays:
        train_essays.append(essay)
        train_essay_feats.append(feats)
    elif essay.name in set_test_essays:
        test_essays.append(essay)
        test_essay_feats.append(feats)

""" DEFINE TAGS """
tag_freq = get_tag_freq(tagged_essays)
freq_tags = list(set((tag for tag, freq in tag_freq.items() if freq >= MIN_TAG_FREQ and regular_tag(tag))))
non_causal  = [t for t in freq_tags if "->" not in t]
only_causal = [t for t in freq_tags if "->" in t]

_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
regular_tags = list(set((t for t in flatten(lst_all_tags) if t[0].isdigit())))

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
sent_output_train_test_tags = list(set(only_causal + CAUSE_TAGS + CAUSAL_REL_TAGS))

assert set(CAUSE_TAGS).issubset(set(sent_input_feat_tags)), "To extract causal relations, we need Causer tags"
# tags to evaluate against
""" END Define Tags """

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """
#fn_create_wd_cls    = lambda : LinearSVC(C=1.0)
fn_create_wd_cls = lambda: LogisticRegression() # C=1, dual = False seems optimal

fn_create_sent_cls  = lambda : LinearSVC(C=1.0)
#fn_create_sent_cls  = lambda : LogisticRegression(dual=True) # C around 1.0 seems pretty optimal

# TD and VD are lists of Essay objects. The sentences are lists
# of featureextractortransformer.Word objects

print "Training Tagging Model"
""" Data Partitioning and Training """
td_feats, td_tags = flatten_to_wordlevel_feat_tags(train_essay_feats)
feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)

td_X = feature_transformer.fit_transform(td_feats)
wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)

""" TRAIN Tagger """
tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, fn_create_wd_cls, wd_train_tags)

print "\nTraining Sentence Model"
""" SENTENCE LEVEL PREDICTIONS FROM STACKING """
sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, train_essay_feats, td_X, wd_td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

""" Train Stacked Classifier """
tag2sent_classifier = train_classifier_per_code(sent_td_xs, sent_td_ys_bycode , fn_create_sent_cls, sent_output_train_test_tags)

""" END TRAINING """

cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)

# TD and VD are lists of Essay objects. The sentences are lists
# of featureextractortransformer.Word objects
print "Running Tagging Model"
""" Data Partitioning and Training """
test_feats, _ = flatten_to_wordlevel_feat_tags(test_essay_feats)

test_x = feature_transformer.transform(test_feats)

""" TEST Tagger """
td_wd_predictions_by_code = test_classifier_per_code(test_x, tag2word_classifier, wd_test_tags)

print "\nRunning Sentence Model"
""" SENTENCE LEVEL PREDICTIONS FROM STACKING """

dummy_wd_td_ys_bytag = defaultdict(lambda : np.asarray([0.0] * test_x.shape[0]))
sent_test_xs, sent_test_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, test_essay_feats, test_x, dummy_wd_td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

""" Test Stack Classifier """
test_sent_predictions_by_code \
    = test_classifier_per_code(sent_test_xs, tag2sent_classifier, sent_output_train_test_tags )

merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)

with open(out_predictions_file, "w+") as f_output_file:
    f_output_file.write("Essay|Sent Number|Processed Sentence|Concept Codes|Predictions\n")
    predictions_to_file(f_output_file, sent_test_ys_bycode, test_sent_predictions_by_code, test_essay_feats, regular_tags + CAUSE_TAGS + CAUSAL_REL_TAGS)
# print results for each code
print out_predictions_file