from sklearn.feature_extraction import DictVectorizer
from searn_parser import SearnModel, PARSE_ACTIONS

class SearnModelSklearnWeighted(SearnModel):

    def __init__(self, feature_extractor, cr_tags, base_learner_fact, beta_decay_fn=lambda b: b - 0.1, positive_val=1, sparse=True):
        super(SearnModelSklearnWeighted, self).__init__(feature_extractor=feature_extractor, cr_tags=cr_tags, base_learner_fact=base_learner_fact, beta_decay_fn=beta_decay_fn, positive_val=positive_val, sparse=sparse)

    def train_parse_models(self, examples):
        models = {}
        self.current_parser_dict_vectorizer = DictVectorizer(sparse=self.sparse)
        xs = self.current_parser_dict_vectorizer.fit_transform(examples.xs)

        for action in PARSE_ACTIONS:

            ys = [1 if i > 0 else 0 for i in examples.get_labels_for(action)]
            weights = examples.get_weights_for(action)

            mdl = self.base_learner_fact()
            mdl.fit(xs, ys, sample_weight=weights)

            models[action] = mdl

        self.current_parser_models = models
        self.parser_models.append(models)
