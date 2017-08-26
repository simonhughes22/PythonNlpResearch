import numpy as np
from sklearn.feature_extraction import DictVectorizer
from searn_parser import SearnModel, PARSE_ACTIONS

class SearnModelCla(SearnModel):
    def __init__(self, feature_extractor, cr_tags, base_learner_fact, beta_decay_fn=lambda b: b - 0.1, positive_val=1, sparse=True):
        super(SearnModelCla, self).__init__(feature_extractor=feature_extractor, cr_tags=cr_tags,
                                                        base_learner_fact=base_learner_fact,
                                                        beta_decay_fn=beta_decay_fn, positive_val=positive_val,
                                                        sparse=sparse)

    def train_parse_models(self, examples):
        models = {}
        self.current_parser_dict_vectorizer = DictVectorizer(sparse=True)
        xs = self.current_parser_dict_vectorizer.fit_transform(examples.xs)

        for action in PARSE_ACTIONS:

            # xgboost needs values in [0,1]
            ys = [1 if i > 0 else 0 for i in examples.get_labels_for(action)]
            weights = examples.get_weights_for(action)

            # the cost matrix has 4 cols - [fp,fn,tp,tn]
            # based on how we compute the costs, the fn cost will be non-zero for positive ground truth
            # else the fp cost will be non-zero. The other 3 cols will be zero

            lst_cost_mat = []
            for lbl, cost in zip(ys, weights):
                fp,fn,tp,tn = 0.05,0.05,0.05,0.05
                if lbl > 0:
                    fn = cost
                else:
                    fp = cost
                lst_cost_mat.append([fp,fn,tp,tn])
            cost_mat = np.asanyarray(lst_cost_mat, dtype=np.float)

            mdl = self.base_learner_fact()
            mdl.fit(xs, ys, cost_mat)

            models[action] = mdl

        self.current_parser_models = models
        self.parser_models.append(models)

    def train_crel_models(self, examples):

        self.current_crel_dict_vectorizer = DictVectorizer(sparse=True)

        xs = self.current_crel_dict_vectorizer.fit_transform(examples.xs)
        ys = examples.get_labels()

        # all costs are equal
        cost_mat = np.ones((len(ys),4),dtype=np.float)
        # Keep this simple as not weighted
        model = self.base_learner_fact()
        model.fit(xs, ys, cost_mat)

        self.current_crel_model = model
        self.crel_models.append(model)