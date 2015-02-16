from featureextractortransformer import FeatureExtractorTransformer
from sent_feats_for_stacking import *
from load_data import load_process_essays_without_annotations

from featureextractionfunctions import *
from wordtagginghelper import *
from predictions_to_file import predictions_to_file
# Classifiers

from model_store import ModelStore
from window_based_tagger_config import get_config
# END Classifiers

import Settings
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

SPARSE_WD_FEATS     = True
SPARSE_SENT_FEATS   = True

MIN_FEAT_FREQ       = 5
CV_FOLDS            = 5

LOOK_BACK           = 0

# construct unique key using settings for pickling

""" PETER - Specify the file paths here to the essay files. """
settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA_Pre_Post_Merged/"
out_predictions_file =              settings.data_directory + "CoralBleaching/Results/predictions.txt"

config = get_config(folder)
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

extractors = [unigram_window_stemmed, biigram_window_stemmed]
feat_config = dict(config.items() + [("extractors", extractors)])

""" LOAD DATA """
tagged_essays = load_process_essays_without_annotations( **config )
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk
feature_extractor = FeatureExtractorTransformer(extractors)

essay_feats = feature_extractor.transform(tagged_essays)
logger.info("Features loaded")

""" LOAD MODELS """
model_store = ModelStore()

feature_transformer = model_store.get_transformer()
tag2word_classifier = model_store.get_tag_2_wd_classifier()
tag2sent_classifier = model_store.get_tag_2_sent_classifier()

""" DEFINE TAGS """
CAUSE_TAGS = ["Causer", "Result", "explicit"]
CAUSAL_REL_TAGS = [CAUSAL_REL, CAUSE_RESULT, RESULT_REL]# + ["explicit"]

""" works best with all the pair-wise causal relation codes """

all_tags = tag2sent_classifier.keys()
regular_tags = [t for t in all_tags if t[0].isdigit()]

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


# Gather metrics
cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag = defaultdict(list), defaultdict(list)

essays_TD = essay_feats

# TD and VD are lists of Essay objects. The sentences are lists
# of featureextractortransformer.Word objects
print "Running Tagging Model"
""" Data Partitioning and Training """
td_feats, _ = flatten_to_wordlevel_feat_tags(essays_TD)

td_X = feature_transformer.fit_transform(td_feats)

""" TEST Tagger """
td_wd_predictions_by_code = test_classifier_per_code(td_X, tag2word_classifier, wd_test_tags)

print "\nRunning Sentence Model"
""" SENTENCE LEVEL PREDICTIONS FROM STACKING """

dummy_wd_td_ys_bytag = defaultdict(lambda : np.asarray([0.0] * td_X.shape[0]))
sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, essays_TD, td_X, dummy_wd_td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

""" Test Stack Classifier """
td_sent_predictions_by_code \
    = test_classifier_per_code(sent_td_xs, tag2sent_classifier, sent_output_train_test_tags )

merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)

with open(out_predictions_file, "w+") as f_output_file:
    f_output_file.write("Essay|Sent Number|Processed Sentence|Concept Codes|Predictions\n")
    predictions_to_file(f_output_file, sent_td_ys_bycode, td_sent_predictions_by_code, essays_TD, regular_tags + CAUSE_TAGS + CAUSAL_REL_TAGS)
# print results for each code
