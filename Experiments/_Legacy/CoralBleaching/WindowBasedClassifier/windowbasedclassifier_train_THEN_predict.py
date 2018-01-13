from featureextractortransformer import FeatureExtractorTransformer
from sent_feats_for_stacking import *
from load_data import load_process_essays, extract_features

from featurevectorizer import FeatureVectorizer
from featureextractionfunctions import *
from CrossValidation import cross_validation
from wordtagginghelper import *
from IterableFP import flatten
# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from window_based_tagger_config import get_config
from model_store import ModelStore
from predictions_to_file import predictions_to_file

# END Classifiers

import Settings
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Create persister (mongo client) - fail fast if mongo service not initialized


# not hashed as don't affect persistence of feature processing
SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling
settings = Settings.Settings()

model_store = ModelStore()

""" PETER - CHANGE THESE FILE PATHS """
folder =                settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"   # Location where the training data is, use EBA_Pre and Post test essays preferably
test_folder=            settings.data_directory + "CoralBleaching/BrattData/Merged/"                # Location where the new essays to tag are located
out_predictions_file =  settings.data_directory + "CoralBleaching/Results/predictions.txt"          # File to dump the predictions to

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
tagged_essays = load_process_essays( **config )
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk
feature_extractor = FeatureExtractorTransformer(extractors)

essay_feats = feature_extractor.transform(tagged_essays)
logger.info("Features loaded")

""" DEFINE TAGS """
_, lst_all_tags = flatten_to_wordlevel_feat_tags(essay_feats)
regular_tags = list(set((t for t in flatten(lst_all_tags) if t[0].isdigit())))

CAUSE_TAGS = ["Causer", "Result", "explicit"]
CAUSAL_REL_TAGS = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]# + ["explicit"]

""" works best with all the pair-wise causal relation codes """
wd_train_tags = regular_tags + CAUSE_TAGS
wd_test_tags  = regular_tags + CAUSE_TAGS

# tags from tagging model used to train the stacked model
sent_input_feat_tags = wd_train_tags
# find interactions between these predicted tags from the word tagger to feed to the sentence tagger
sent_input_interaction_tags = wd_train_tags
# tags to train (as output) for the sentence based classifier
sent_output_train_test_tags = list(set(regular_tags + CAUSE_TAGS + CAUSAL_REL_TAGS))

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

#TODO Parallelize
essays_TD = essay_feats

# TD and VD are lists of Essay objects. The sentences are lists
# of featureextractortransformer.Word objects

print "Training Tagging Model"
""" Data Partitioning and Training """
td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)

td_X = feature_transformer.fit_transform(td_feats)
wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)

""" TRAIN Tagger """
tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, fn_create_wd_cls, wd_train_tags)

print "\nTraining Sentence Model"
""" SENTENCE LEVEL PREDICTIONS FROM STACKING """
sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, essays_TD, td_X, wd_td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

""" Train Stacked Classifier """
tag2sent_classifier = train_classifier_per_code(sent_td_xs, sent_td_ys_bycode , fn_create_sent_cls, sent_output_train_test_tags)


""" END TRAINING """

test_config = get_config(test_folder)

test_tagged_essays = load_process_essays(**test_config)
test_essay_feats = feature_extractor.transform(test_tagged_essays)

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