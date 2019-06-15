from collections import defaultdict
import pickle
import numpy as np

class StructuredPerceptron(object):
    '''A structured perceptron, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    '''

    def __init__(self, learning_rate=0.3, max_update_items=1, initial_weight=1):
        # Each feature gets its own weight
        # needs to be non zero otherwise first
        self.initial_weight = initial_weight
        self.weights = defaultdict(lambda : initial_weight)
        self.learning_rate = learning_rate
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(float)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0
        # how many items do we use to update the weights?
        self.max_update_items = max_update_items
        self.averaged = False

    def num_feats(self):
        return len(self.weights)

    def clone(self):
        cloney = StructuredPerceptron(self.learning_rate)
        cloney.weights.update(self.weights)
        cloney._totals.update(self._totals)
        cloney._tstamps.update(self._tstamps)
        cloney.i = self.i
        cloney.averaged = self.averaged
        return cloney

    def rank(self, features_array, existence_check=True):
        '''Dot-product the features and current weights and return the best label.'''
        scores2index = {}
        for i, feats in enumerate(features_array):
            scores2index[i] = self.decision_function(feats, existence_check)
        # return a ranking of the scores, by best to worse
        return [ix for ix, score in sorted(scores2index.items(), key=lambda tpl: -tpl[-1])]

    def decision_function(self, features, existence_check=True):
        '''Dot-product the features and current weights and return the score.'''
        score = 0.0
        for feat, value in features.items():
            if value == 0:
                continue
            if existence_check and feat not in self.weights:
                continue

            score += self.weights[feat] * value
            assert np.isnan(score) == False, \
                "decision function value is nan for feat: {feat} and val: {val} and weight: {weight}".format(
                feat=feat, val=value, weight=self.weights[feat])
        return score

    def train(self, best_feats, other_feats_array):
        feats_array = [best_feats] + list(other_feats_array)
        ixs = self.rank(feats_array, existence_check=False)

        # go thru up to |max_update_items| items ranked above the best, and update the weights
        best_ix = ixs[0]
        if best_ix != 0:
            for rank, ix in enumerate(ixs):
                # don't update items ranked below the best parse
                if ix == 0 or rank >= self.max_update_items:
                    break

                self.update(best_feats=best_feats, highest_ranked_feats=feats_array[ix])

    def __upd_feat__(self, feat, val):
        w = self.weights[feat]
        # update the totals by the number of timestamps the current value has survived * val
        self._totals[feat] += (self.i - self._tstamps[feat]) * w
        # store latest update timestamp
        self._tstamps[feat] = self.i
        # finally, update the current weight
        self.weights[feat] = w + (self.learning_rate * val)

    def update(self, best_feats, highest_ranked_feats):
        '''Update the feature weights.'''

        self.i += 1
        for feat, weight in self.weights.items():
            val = best_feats[feat] - highest_ranked_feats[feat]
            self.__upd_feat__(feat, val)
        return None

    def average_weights(self):
        # This operation is not 100% idempotent, so prevent doing it multiple times
        assert self.averaged == False, "Weights already averaged"
        '''Average weights from all iterations.'''
        new_feat_weights = defaultdict(float)
        for feat, weight in self.weights.items():
            total = self._totals[feat]
            total += (self.i - self._tstamps[feat]) * weight
            averaged = round(total / float(self.i), 5)
            if averaged != 0.0:
                new_feat_weights[feat] = averaged
        self.weights = new_feat_weights
        self.averaged = True
        return None

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None


class CostSensitiveStructuredPerceptron(StructuredPerceptron):
    def __init__(self, *args, **kwargs):
        super(CostSensitiveStructuredPerceptron, self).__init__(*args, **kwargs)

    def train(self, best_feats, other_feats_array, other_costs_array):

        feats_array = [best_feats] + list(other_feats_array)
        costs_array = [0] + other_costs_array
        ixs = self.rank(feats_array, existence_check=False)

        # go thru up to |max_update_items| items ranked above the best, and update the weights
        best_ix = ixs[0]
        if best_ix != 0:
            for rank, ix in enumerate(ixs):
                # don't update items ranked below the best parse
                if ix == 0 or rank >= self.max_update_items:
                    break

                self.update(best_feats=best_feats,
                            highest_ranked_feats=feats_array[ix], highest_ranked_cost=costs_array[ix])

    def clone(self):
        p = CostSensitiveStructuredPerceptron(self.learning_rate)
        p.weights.update(self.weights)
        p._totals.update(self._totals)
        p._tstamps.update(self._tstamps)
        p.i = self.i
        return p

    def update(self, best_feats, highest_ranked_feats, highest_ranked_cost):
        '''Update the feature weights.'''

        self.i += 1
        for feat, weight in self.weights.items():
            val = (best_feats[feat] - highest_ranked_feats[feat]) * highest_ranked_cost
            self.__upd_feat__(feat, val)
        return None


if __name__ == "__main__":

    p = CostSensitiveStructuredPerceptron(learning_rate=0.1)
    best = defaultdict(float)
    best.update({ "a": 1, "b": 2})

    rest = defaultdict(float)
    rest.update({"a": -1, "b": 3, "c": 4})

    p.train(best, [rest])
    pass
