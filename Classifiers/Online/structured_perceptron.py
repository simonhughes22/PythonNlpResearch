"""
Structured perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""
from collections import defaultdict
import pickle

class StructuredPerceptron(object):
    '''A structured perceptron, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    '''

    def __init__(self, learning_rate):
        # Each feature gets its own weight
        self.weights = defaultdict(float)
        self.learning_rate = learning_rate
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def rank(self, features_array):
        '''Dot-product the features and current weights and return the best label.'''
        scores2index = {}
        for i, feats in enumerate(features_array):
            scores2index[i] = self.decision_function(feats)
        # return a ranking of the scores, by best to worse

        return [ix for ix, score in sorted(scores2index.items(), key=lambda tpl: -tpl[-1])]

    def train(self, best_feats, other_feats_array):
        best_ix = self.rank([best_feats] + list(other_feats_array))
        if best_ix != 0:
            predicted_feats = other_feats_array[best_ix - 1]
            self.update(best_feats=best_feats, highest_ranked_feats=predicted_feats)

    def decision_function(self, features):
        '''Dot-product the features and current weights and return the score.'''
        score = 0.0
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            score += self.weights[feat] * value
        return score

    def update(self, best_feats, highest_ranked_feats):
        '''Update the feature weights.'''

        # TODO - weight the weight update by the difference in errors
        def upd_feat(feat, val):
            w = self.weights[feat]
            # update the totals by the number of timestamps the current value has survived * val
            self._totals[feat] += (self.i - self._tstamps[feat]) * w
            # store latest update timestamp
            self._tstamps[feat] = self.i
            # finally, update the current weight
            self.weights[feat] = w + (self.learning_rate * val)

        self.i += 1
        for feat, weight in self.weights.items():
            val = best_feats[feat] - highest_ranked_feats[feat]
            upd_feat(feat, val)
        return None

    def average_weights(self):
        '''Average weights from all iterations.'''
        new_feat_weights = defaultdict(float)
        for feat, weight in self.weights.items():
            total = self._totals[feat]
            total += (self.i - self._tstamps[feat]) * weight
            averaged = round(total / float(self.i), 5)
            if averaged != 0.0:
                new_feat_weights[feat] = averaged
        self.weights = new_feat_weights
        return None

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None
