from featurevectorizer import FeatureVectorizer
from searn_parser_template_features import SearnModelTemplateFeatures
from shift_reduce_helper import PARSE_ACTIONS, allowed_action

class SearnModelTemplateFeaturesRegression(SearnModelTemplateFeatures):
    def __init__(self, ngram_extractor, feature_extractor, cost_function, min_feature_freq, cr_tags,
                     base_learner_fact, crel_learner_fact,
                     beta_decay_fn=lambda b: b - 0.1, positive_val=1, sparse=True, log_fn=lambda s: print(s)):

        super(SearnModelTemplateFeaturesRegression, self).__init__(ngram_extractor=ngram_extractor,
                                                                   feature_extractor=feature_extractor,
                                                                   cost_function=cost_function,
                                                                   min_feature_freq=min_feature_freq,
                                                                   cr_tags=cr_tags,
                                                                   base_learner_fact=base_learner_fact,
                                                                   beta_decay_fn=beta_decay_fn,
                                                                   positive_val=positive_val,
                                                                   sparse=sparse)
        self.crel_learner_fact = crel_learner_fact

    def train_parse_models(self, examples):
        models = {}
        self.current_parser_feat_vectorizer = FeatureVectorizer(min_feature_freq=self.min_feature_freq,
                                                                sparse=self.sparse)
        xs = self.current_parser_feat_vectorizer.fit_transform(examples.xs)

        for action in PARSE_ACTIONS:
            # positive examples have negative cost, negative examples have positive cost
            lbls = [1 if i > 0 else -1 for i in examples.get_labels_for(action)]    # type: List[int]
            costs = examples.get_weights_for(action)                              # type: List[float]

            ys = [lbl * cost for (lbl,cost) in zip(lbls, costs)]

            mdl = self.base_learner_fact()
            mdl.fit(xs, ys, sample_weight=costs)

            models[action] = mdl

        self.current_parser_models = models
        self.parser_models.append(models)

    def predict_parse_action(self, feats, tos):
        xs = self.current_parser_feat_vectorizer.transform(feats)
        prob_by_label = {}
        for action in PARSE_ACTIONS:
            if not allowed_action(action, tos):
                continue

            prob_by_label[action] = self.current_parser_models[action].predict(xs)[0]

        # Get label with the lowest cost
        min_act, min_val = min(prob_by_label.items(), key=lambda tpl: tpl[1])
        return min_act

