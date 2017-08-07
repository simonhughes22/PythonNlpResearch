# coding: utf-8

import pymongo

#client = pymongo.MongoClient()
#db = client.metrics

from window_based_tagger_config import get_config
from results_procesor import ResultsProcessor

import pickle
from collections import defaultdict
from Settings import Settings

CV_FOLDS = 5
DEV_SPLIT = 0.1

settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder = root_folder + "Training" + "/"
training_pickled = settings.data_directory + "CoralBleaching/Thesis_Dataset/training.pl"
predictions_folder = root_folder + "Predictions/Bi-LSTM-4-SEARN/"

config = get_config(training_folder)
processor = ResultsProcessor()

# with open(training_pickled, "rb+") as f:
#     tagged_essays = pickle.load(f)
# len(tagged_essays)
# ## Load Training Data - Essays Tagged with Codes By a Bi-Directional RNN

# In[5]:

import dill

fname = predictions_folder + "essays_train_bi_directional-True_hidden_size-256_merge_mode-sum_num_rnns-2_use_pretrained_embedding-True.dill"
with open(fname, "rb") as f:
    pred_tagged_essays = dill.load(f)

# len(tagged_essays),
print("Number of pred tagged essasy %i" % len(pred_tagged_essays) ) # should be 902

# In[6]:

import datetime, logging

print("Started at: " + str(datetime.datetime.now()))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# In[7]:

CAUSER = "Causer"
RESULT = "Result"
EXPLICIT = "explicit"
CAUSER_EXPLICIT = "Causer_Explicit"
EXPLICIT_RESULT = "Explicit_Result"
CAUSER_EXPLICIT_RESULT = "Causer_Explicit_Result"
CAUSER_RESULT = "Causer_Result"

# In[8]:

tag_freq = defaultdict(int)
unique_words = set()
for essay in pred_tagged_essays:
    for sentence in essay.sentences:
        for word, tags in sentence:
            unique_words.add(word)
            for tag in tags:
                tag_freq[tag] += 1

# TODO - don't ignore Anaphor, other and rhetoricals here
cr_tags = list(
    (t for t in tag_freq.keys() if ("->" in t) and not "Anaphor" in t and not "other" in t and not "rhetorical" in t))
#Change to include explicit
regular_tags = set((t for t in tag_freq.keys() if ( "->" not in t) and (t == "explicit" or t[0].isdigit())))
vtags = set(regular_tags)

assert "explicit" in vtags, "explicit should be in the regular tags"

# In[14]:

from parser_feature_extractor import FeatureExtractor, bag_of_word_extractor, bag_of_word_plus_tag_extractor

feat_extractor = FeatureExtractor([
    bag_of_word_extractor,
    bag_of_word_plus_tag_extractor,
])

# In[49]:

from sklearn.linear_model import LogisticRegression
from searn_parser import SearnModel, EMPTY_TAG
import numpy as np

# Replace predicted tags with actual to test impact on recall
for essay_ix, essay in enumerate(pred_tagged_essays):
    act_tags = []
    for sent_ix, taggged_sentence in enumerate(essay.sentences):
        sent_tags = []
        act_tags.append(sent_tags)
        for wd, tags in taggged_sentence:
            rtags = list(vtags.intersection(tags))
            if len(rtags) == 0:
                sent_tags.append(EMPTY_TAG)
            else:
                np.random.shuffle(rtags)
                sent_tags.append(rtags[0])

    essay.pred_tagged_sentences = act_tags

#parse_model = SearnModel(feat_extractor, cr_tags, base_learner_fact=LogisticRegression, beta_decay_fn=lambda beta: beta - 0.3)
parse_model = SearnModel(feat_extractor, cr_tags, base_learner_fact=LogisticRegression, beta_decay_fn=lambda beta: beta)
parse_model.train(pred_tagged_essays, 12)

## TODO
#- Re-train tagging model, adding tags where reg tag is missing but is included in a causer or result tag.
#- Also include explicit in the predicted tags.
#- Need to handle relations where same code -> same code

#TODO * Need to make sure the tagger tags EXCPLICT tags. These can then be skipped by the parser, but will be included in the features used to train the parser and taggger. Do we want to train a separate tagger that determines if a tagged word is a cause, explict or result. That will then resolve the direction of the relation?
#TODO - recall is v low on training data. Test it with perfect tagging predictions

#TODO Issues
# 1. Unsupported relations
# 2. Tagging model needs to tag causer:num and result:num too as tags, as well as explicits
# 3. Can't handle same tag to same tag
# 4. Can't handle same relation in both directions (e.g. if code is repeated)