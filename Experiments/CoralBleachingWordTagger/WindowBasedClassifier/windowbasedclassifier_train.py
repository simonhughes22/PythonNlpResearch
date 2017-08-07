import logging

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import Settings
from IterableFP import flatten
from featureextractionfunctions import *
from featureextractortransformer import FeatureExtractorTransformer
from featurevectorizer import FeatureVectorizer
from load_data import load_process_essays
from model_store import ModelStore
from sent_feats_for_stacking import *
from window_based_tagger_config import get_config
from wordtagginghelper import *

# END Classifiers
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

folder =                            settings.data_directory + "CoralBleaching/BrattData/EBA1415_Merged/"
config = get_config(folder)

""" FEATURE EXTRACTION """
offset = int((config["window_size"] - 1) / 2)

unigram_window_stemmed = fact_extract_positional_word_features_stemmed(offset)
biigram_window_stemmed = fact_extract_ngram_features_stemmed(offset, 2)

extractors = [unigram_window_stemmed, biigram_window_stemmed]
feat_config = dict(list(config.items()) + [("extractors", extractors)])

""" LOAD DATA """
tagged_essays = load_process_essays( **config )
logger.info("Essays loaded")
# most params below exist ONLY for the purposes of the hashing to and from disk
feature_extractor = FeatureExtractorTransformer(extractors)

essay_feats = feature_extractor.transform(tagged_essays)
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

# tags from tagging model used to train the stacked model
sent_input_feat_tags = list(set(freq_tags + CAUSE_TAGS))
# find interactions between these predicted tags from the word tagger to feed to the sentence tagger
sent_input_interaction_tags = list(set(non_causal + CAUSE_TAGS))
# tags to train (as output) for the sentence based classifier
sent_output_train_test_tags = list(set(all_tags + CAUSE_TAGS + CAUSAL_REL_TAGS))

assert set(CAUSE_TAGS).issubset(set(sent_input_feat_tags)), "To extract causal relations, we need Causer tags"# tags to evaluate against

""" CLASSIFIERS """
""" Log Reg + Log Reg is best!!! """
fn_create_wd_cls = lambda: LogisticRegression() # C=1, dual = False seems optimal
fn_create_sent_cls  = lambda : LogisticRegression(dual=True) # C around 1.0 seems pretty optimal
# NOTE - GBT is stochastic in the SPLITS, and so you will get non-deterministic results

if type(fn_create_sent_cls()) == GradientBoostingClassifier:
    SPARSE_SENT_FEATS = False

#TODO Parallelize
essays_TD = essay_feats

# TD and VD are lists of Essay objects. The sentences are lists
# of featureextractortransformer.Word objects

print("Training Tagging Model")
""" Data Partitioning and Training """
td_feats, td_tags = flatten_to_wordlevel_feat_tags(essays_TD)
feature_transformer = FeatureVectorizer(min_feature_freq=MIN_FEAT_FREQ, sparse=SPARSE_WD_FEATS)

td_X = feature_transformer.fit_transform(td_feats)
wd_td_ys_bytag = get_wordlevel_ys_by_code(td_tags, wd_train_tags)

""" TRAIN Tagger """
tag2word_classifier = train_classifier_per_code(td_X, wd_td_ys_bytag, fn_create_wd_cls, wd_train_tags)

print("\nTraining Sentence Model")
""" SENTENCE LEVEL PREDICTIONS FROM STACKING """
sent_td_xs, sent_td_ys_bycode = get_sent_feature_for_stacking_from_tagging_model(sent_input_feat_tags, sent_input_interaction_tags, essays_TD, td_X, wd_td_ys_bytag, tag2word_classifier, SPARSE_SENT_FEATS, LOOK_BACK)

""" Train Stacked Classifier """
tag2sent_classifier = train_classifier_per_code(sent_td_xs, sent_td_ys_bycode , fn_create_sent_cls, sent_output_train_test_tags)

""" Persist Models """
model_store.store(feature_transformer, tag2word_classifier, tag2sent_classifier)