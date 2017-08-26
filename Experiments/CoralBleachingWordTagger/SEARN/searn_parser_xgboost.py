from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

from NgramGenerator import compute_ngrams
from Rpfa import micro_rpfa
from oracle import Oracle
from parser import Parser
from results_procesor import ResultsProcessor
from searn_parser import SearnModel, PARSE_ACTIONS
from shift_reduce_helper import *
from stack import Stack
from weighted_examples import WeightedExamples
import numpy as np
import string
import xgboost as xgb

class SearnModelXgBoost(SearnModel):
    def __init__(self, feature_extractor, cr_tags, beta_decay_fn=lambda b: b - 0.1, positive_val=1):
        super(SearnModelXgBoost, self).__init__(feature_extractor=feature_extractor, cr_tags=cr_tags,
                                                base_learner_fact=xgb.XGBClassifier,
                                                beta_decay_fn=beta_decay_fn, positive_val=positive_val,
                                                sparse=True)
    def train_parse_models(self, examples):
        models = {}
        self.current_parser_dict_vectorizer = DictVectorizer(sparse=self.sparse)
        xs = self.current_parser_dict_vectorizer.fit_transform(examples.xs)

        for action in PARSE_ACTIONS:

            # xgboost needs values in [0,1]
            ys = [1 if i > 0 else 0 for i in examples.get_labels_for(action)]
            weights = examples.get_weights_for(action)
            # TODO - maximize F1 score here in training the tree?
            #dtrain = xgb.DMatrix(xs, label=ys, weight=weights, silent=True)
            # params = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
            #params = {'silent': 1, 'objective': 'binary:logistic'}
            #num_round = 10
            # http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
            #mdl = xgb.train( params, dtrain=dtrain, num_boost_round=num_round, verbose_eval=False)

            mdl = self.base_learner_fact()
            mdl.fit(xs, ys, sample_weight=weights)

            models[action] = mdl

        self.current_parser_models = models
        self.parser_models.append(models)

    def predict_parse_action(self, feats, tos):
        xs = self.current_parser_dict_vectorizer.transform(feats)
        #dpred = xgb.DMatrix(xs)

        prob_by_label = {}
        for action in PARSE_ACTIONS:
            if not self.allowed_action(action, tos):
                continue

            # NOTE: as it was trained with logistic fn, it actually predicts a probability here
            #prob_by_label[action] = self.current_parser_models[action].predict(dpred, output_margin=False)[0]
            prob_by_label[action] = self.current_parser_models[action].predict_proba(xs, output_margin=False)[0]

        max_act, max_prob = max(prob_by_label.items(), key=lambda tpl: tpl[1])
        return max_act
