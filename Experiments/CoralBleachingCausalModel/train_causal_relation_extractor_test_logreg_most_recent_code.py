# coding: utf-8
import datetime
import logging

import dill
import pymongo
from sklearn.linear_model import LogisticRegression

from Settings import Settings
from cost_functions import *
from crel_helper import get_cr_tags
from function_helpers import get_function_names, get_functions_by_name
from searn_parser_breadth_first import SearnModelBreadthFirst
from template_feature_extractor import *
from window_based_tagger_config import get_config

# Logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()

# Mongo connection
client = pymongo.MongoClient()
db = client.metrics

# Data Set Partition
CV_FOLDS = 5
MIN_FEAT_FREQ = 5

# Global settings
logger.info("Started at: " + str(datetime.datetime.now()))
settings = Settings()
root_folder = settings.data_directory + "CoralBleaching/Thesis_Dataset/"
training_folder = root_folder + "Training" + "/"
test_folder = root_folder + "Test" + "/"
# NOTE: These predictions are generated from the "./notebooks/SEARN/Keras - Train Tagger and Save CV Predictions For Word Tags.ipynb" notebook
# used as inputs to parsing model
coref_root = root_folder + "CoReference/"
coref_output_folder = coref_root + "CRel/"

config = get_config(training_folder)

train_fname = coref_output_folder + "training_crel_anatagged_essays_most_recent_code.dill"
with open(train_fname, "rb") as f:
    pred_tagged_essays_train = dill.load(f)

test_fname = coref_output_folder + "test_crel_anatagged_essays_most_recent_code.dill"
with open(test_fname, "rb") as f:
    pred_tagged_essays_test = dill.load(f)

logger.info("Number of pred tagged essays %i" % len(pred_tagged_essays_train))  # should be 902

cr_tags = get_cr_tags(train_tagged_essays=pred_tagged_essays_train, tag_essays_test=pred_tagged_essays_test)


def model_train_predict(essays_TD, essays_VD, extractor_names, cost_function_name, ngrams, stemmed, beta, max_epochs):

    #logger.info("\tModei={model}".format(model=str(BASE_LEARNER_FACT())))

    extractors = get_functions_by_name(extractor_names, all_extractor_fns)
    # get single cost function
    cost_fn = get_functions_by_name([cost_function_name], all_cost_functions)[0]
    assert cost_fn is not None, "Cost function look up failed"
    # Ensure all extractors located
    assert len(extractors) == len(extractor_names), "number of extractor functions does not match the number of names"

    template_feature_extractor = NonLocalTemplateFeatureExtractor(extractors=extractors)
    if stemmed:
        ngram_extractor = NgramExtractorStemmed(max_ngram_len=ngrams)
    else:
        ngram_extractor = NgramExtractor(max_ngram_len=ngrams)

    parse_model = SearnModelBreadthFirst(feature_extractor=template_feature_extractor,
                                       cost_function=cost_fn,
                                       min_feature_freq=MIN_FEAT_FREQ,
                                       ngram_extractor=ngram_extractor, cr_tags=cr_tags,
                                       base_learner_fact=BASE_LEARNER_FACT,
                                       beta=beta,
                                       # log_fn=lambda s: print(s))
                                       log_fn=lambda s: None)

    parse_model.train(essays_TD, max_epochs=max_epochs)

    num_feats = template_feature_extractor.num_features()

    sent_td_ys_bycode = parse_model.get_label_data(essays_TD)
    sent_vd_ys_bycode = parse_model.get_label_data(essays_VD)

    sent_td_pred_ys_bycode = parse_model.predict(essays_TD)
    sent_vd_pred_ys_bycode = parse_model.predict(essays_VD)

    return parse_model, num_feats, sent_td_ys_bycode, sent_vd_ys_bycode, sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode


LINE_WIDTH = 80

# other settings
DOWN_SAMPLE_RATE = 1.0  # For faster smoke testing the algorithm
BASE_LEARNER_FACT = None
COLLECTION_PREFIX = "TEST_CR_CB_SHIFT_REDUCE_PARSER_TEMPLATED_MOST_RECENT_CODE"

# some of the other extractors aren't functional if the system isn't able to do a basic parse
# so the base extractors are the MVP for getting to a basic parser, then additional 'meta' parse
# features from all_extractors can be included
base_extractors = [
    single_words,
    word_pairs,
    three_words,
    between_word_features
]

all_extractor_fns = base_extractors + [
    word_distance,
    valency,
    unigrams,
    third_order,
    label_set,
    size_features
]

all_cost_functions = [
    micro_f1_cost,
    micro_f1_cost_squared,
    micro_f1_cost_plusone,
    micro_f1_cost_plusepsilon,
    binary_cost,
    inverse_micro_f1_cost,
    uniform_cost
]

all_extractor_fn_names = get_function_names(all_extractor_fns)
base_extractor_fn_names = get_function_names(base_extractors)
all_cost_fn_names = get_function_names(all_cost_functions)


ngrams = 1
stemmed = True
cost_function_name = micro_f1_cost_plusepsilon.__name__
dual = True
fit_intercept = True
beta = 0.5
max_epochs = 2
C = 0.5
penalty = "l2"

# Note these also differ for SC dataset
BASE_LEARNER_FACT = lambda : LogisticRegression(dual=dual, C=C, penalty=penalty, fit_intercept=fit_intercept)
best_extractor_names = ['single_words', 'between_word_features', 'label_set',
                                    'three_words', 'third_order', 'unigrams'] # type: List[str]

logger.info("Training Model")
train_result = model_train_predict(pred_tagged_essays_train, pred_tagged_essays_test, best_extractor_names, cost_function_name, ngrams, stemmed, beta, max_epochs)
parse_model, num_feats, sent_td_ys_bycode, sent_vd_ys_bycode, sent_td_pred_ys_bycode, sent_vd_pred_ys_bycode = train_result

logger.info("Predicting")
parses = []

def get_crels(parse):
    crels = set()
    p = parse
    while p:
        if p.relations:
            crels.update(p.relations)
        p = p.parent_action
    return crels

for eix, essay in enumerate(pred_tagged_essays_test):
    for sent_ix, taggged_sentence in enumerate(essay.sentences):
        predicted_tags = essay.pred_tagged_sentences[sent_ix]
        unq_ptags = set(predicted_tags)
        if len(unq_ptags) > 2:
            pred_crels = parse_model.predict_sentence(tagged_sentence=taggged_sentence, predicted_tags=predicted_tags)

            pred_parses = parse_model.generate_all_potential_parses_for_sentence(
                tagged_sentence=taggged_sentence, predicted_tags=predicted_tags, top_n=100)
            parses.append((eix, sent_ix, pred_parses))

            assert pred_crels == get_crels(pred_parses[0]), "Parser miss match"
            print("tags: ", len(unq_ptags), "\tparses:", len(pred_parses))

logger.info("Fimnished")