from featurevectorizer import FeatureVectorizer
from searn_parser_logreg import SearnModelTemplateFeatures
from shift_reduce_helper import PARSE_ACTIONS, allowed_action

import numpy as np

class SearnModelTemplateFeaturesMultinomialLogisticRegression(SearnModelTemplateFeatures):
    def __init__(self, ngram_extractor, feature_extractor, cost_function, min_feature_freq, cr_tags,
                 base_learner_fact, crel_learner_fact,
                 beta_decay_fn=lambda b: b - 0.1, positive_val=1, sparse=True, log_fn=lambda s: print(s)):

        super(SearnModelTemplateFeaturesMultinomialLogisticRegression, self).__init__(ngram_extractor=ngram_extractor,
                                                                                      feature_extractor=feature_extractor,
                                                                                      cost_function=cost_function,
                                                                                      min_feature_freq=min_feature_freq,
                                                                                      cr_tags=cr_tags,
                                                                                      base_learner_fact=base_learner_fact,

                                                                                      beta_decay_fn=beta_decay_fn,
                                                                                      positive_val=positive_val,
                                                                                      sparse=sparse,
                                                                                      log_fn=log_fn)
        self.crel_learner_fact = crel_learner_fact

    def train_parse_models(self, examples):
        self.current_parser_feat_vectorizer = FeatureVectorizer(min_feature_freq=self.min_feature_freq,
                                                                sparse=self.sparse)
        xs = self.current_parser_feat_vectorizer.fit_transform(examples.xs)
        ys = examples.get_labels()

        weights = []

        for ix in range(xs.shape[0]):
            costs_by_action = {}
            gold_action = ys[ix]
            gold_action_wt = 0
            for action in PARSE_ACTIONS:
                cost = examples.get_weights_for(action)[ix]
                if action == gold_action:
                    gold_action_wt = cost
                else:
                    costs_by_action[action] = cost

            worse_action, worse_cost = max(costs_by_action.items(), key= lambda tpl: tpl[1])
            assert gold_action_wt >=0 and worse_cost >= 0, "Costs should be non negative"
            # Weight of example is the difference between the best action and the worse action
            # as both are positive, we simply add them up
            weight = gold_action_wt + worse_cost
            weights.append(weight)

        mdl = self.base_learner_fact()
        mdl.fit(xs, ys, sample_weight=weights)

        #cost = examples.get_weights_for(action)[ix]
        self.current_parser_models = mdl
        self.parser_models.append(mdl)

    def predict_parse_action(self, feats, tos):
        model = self.current_parser_models

        xs = self.current_parser_feat_vectorizer.transform(feats)
        # get first row, as just looking at one data point
        ys_probs = model.predict_proba(xs)[0]

        prob_by_label = {}
        for action, prob in zip(model.classes_, ys_probs):
            if not allowed_action(action, tos):
                continue
            prob_by_label[action] = prob

        items = list(prob_by_label.items())
        # randomize order so that max returns different items in the case of a tie
        np.random.shuffle(items)
        max_act, max_prob = max(items, key=lambda tpl: tpl[1])
        return max_act
