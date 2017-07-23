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
regular_tags = list((t for t in tag_freq.keys() if t[0].isdigit()))

vtags = set(regular_tags)

# In[14]:

from parser_feature_extractor import FeatureExtractor, bag_of_word_extractor, bag_of_word_plus_tag_extractor

feat_extractor = FeatureExtractor([
    bag_of_word_extractor,
    bag_of_word_plus_tag_extractor,
])

# In[49]:

from sklearn.linear_model import LogisticRegression
from searn_parser import SearnModel

parse_model = SearnModel(feat_extractor, cr_tags, base_learner_fact=LogisticRegression)
parse_model.train(pred_tagged_essays, 2)

#TODO * Need to make sure the tagger tags EXCPLICT tags. These can then be skipped by the parser, but will be included in the features used to train the parser and taggger. Do we want to train a separate tagger that determines if a tagged word is a cause, explict or result. That will then resolve the direction of the relation?
