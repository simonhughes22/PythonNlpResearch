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
from sent_feats_for_stacking import get_sent_tags_from_word_tags
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

MIN_FEAT_FREQ       = 5        # 5 best so far
CV_FOLDS            = 5

MIN_TAG_FREQ        = 5
LOOK_BACK           = 0     # how many sentences to look back when predicting tags
# end not hashed

# construct unique key using settings for pickling

settings = Settings.Settings()
folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
processed_essay_filename_prefix =   settings.data_directory + "CoralBleaching/BrattData/Pickled/essays_proc_pickled_"
features_filename_prefix =          settings.data_directory + "CoralBleaching/BrattData/Pickled/feats_pickled_"

out_metrics_file     =              settings.data_directory + "CoralBleaching/Results/metrics.txt"
out_predictions_file =              settings.data_directory + "CoralBleaching/Results/predictions.txt"

config = get_config(folder)

""" FEATURE EXTRACTION """
offset = (config["window_size"] - 1) / 2

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

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
#wd_train_tags = list(set(all_tags + CAUSE_TAGS))
#wd_test_tags  = list(set(all_tags + CAUSE_TAGS))
wd_train_tags = list(set(only_causal))
wd_test_tags  = list(set(only_causal))

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
    wd_vd_ys_bytag = get_wordlevel_ys_by_code(vd_tags, wd_test_tags)

    """ TRAIN Tagger """
    tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, fn_create_wd_cls, wd_train_tags)

    """ TEST Tagger """
    td_wd_predictions_by_code = test_classifier_per_code(td_X, tag2word_classifier, wd_test_tags)
    vd_wd_predictions_by_code = test_classifier_per_code(vd_X, tag2word_classifier, wd_test_tags)

    print "Aggregate word predictions to sentence level"
    td_sent_predictions_by_code = get_sent_tags_from_word_tags(essays_TD, td_wd_predictions_by_code)
    vd_sent_predictions_by_code = get_sent_tags_from_word_tags(essays_VD, vd_wd_predictions_by_code)

    """ get sentence level labels """
    sent_td_ys_bycode = get_sent_tags_from_word_tags(essays_TD, wd_td_ys_bytag)
    sent_vd_ys_bycode = get_sent_tags_from_word_tags(essays_VD, wd_vd_ys_bytag)

    merge_dictionaries(wd_td_ys_bytag, cv_wd_td_ys_by_tag)
    merge_dictionaries(wd_vd_ys_bytag, cv_wd_vd_ys_by_tag)
    merge_dictionaries(td_wd_predictions_by_code, cv_wd_td_predictions_by_tag)
    merge_dictionaries(vd_wd_predictions_by_code, cv_wd_vd_predictions_by_tag)

    merge_dictionaries(sent_td_ys_bycode, cv_sent_td_ys_by_tag)
    merge_dictionaries(sent_vd_ys_bycode, cv_sent_vd_ys_by_tag)
    merge_dictionaries(td_sent_predictions_by_code, cv_sent_td_predictions_by_tag)
    merge_dictionaries(vd_sent_predictions_by_code, cv_sent_vd_predictions_by_tag)

    predictions_to_file(f_output_file, sent_vd_ys_bycode, vd_sent_predictions_by_code, essays_VD, codes=wd_test_tags)

f_output_file.close()
# print results for each code
logger.info("Training completed")

""" Persist Results to Mongo DB """

wd_algo   = str(fn_create_wd_cls())
sent_algo = str(fn_create_sent_cls())

SUFFIX = "_CAUSE_EFFECT_LBLS"
CB_TAGGING_TD, CB_TAGGING_VD, CB_SENT_TD, CB_SENT_VD = "CB_TAGGING_TD" + SUFFIX, "CB_TAGGING_VD" + SUFFIX, "CB_SENT_TD" + SUFFIX, "CB_SENT_VD" + SUFFIX
parameters = dict(config)
parameters["extractors"] = map(lambda fn: fn.func_name, extractors)
parameters["min_feat_freq"] = MIN_FEAT_FREQ
parameters["no_stacking"] = True

wd_td_objectid = processor.persist_results(CB_TAGGING_TD, cv_wd_td_ys_by_tag, cv_wd_td_predictions_by_tag, parameters, wd_algo)
wd_vd_objectid = processor.persist_results(CB_TAGGING_VD, cv_wd_vd_ys_by_tag, cv_wd_vd_predictions_by_tag, parameters, wd_algo)

sent_td_objectid = processor.persist_results(CB_SENT_TD, cv_sent_td_ys_by_tag, cv_sent_td_predictions_by_tag, parameters, sent_algo, tagger_id=wd_td_objectid)
sent_vd_objectid = processor.persist_results(CB_SENT_VD, cv_sent_vd_ys_by_tag, cv_sent_vd_predictions_by_tag, parameters, sent_algo, tagger_id=wd_vd_objectid)

print processor.results_to_string(wd_td_objectid,   CB_TAGGING_TD,  wd_vd_objectid,     CB_TAGGING_VD,  "TAGGING")
print processor.results_to_string(sent_td_objectid, CB_SENT_TD,     sent_vd_objectid,   CB_SENT_VD,     "SENTENCE")
logger.info("Results Processed")
