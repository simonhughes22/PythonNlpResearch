from collections import defaultdict
import pickle

class ALMA(object):
    ''' ALMA Algorithm - see pages 175-176 in my structured learning book
    '''

    def __init__(self, features, alpha=1.0, B=None, C=2**0.5):
        # Each feature gets its own weight
        # needs to be non zero otherwise first
        if B is None:
            B = 1/alpha
        self.C = C
        self.B = B
        self.alpha = alpha
        self.features = features
        self.weights = self.proj(dict([(f,1) for f in features]))
        self.k = 1

    def l2_norm(self, weights):
        return sum((v ** 2 for v in weights.values())) ** 0.5

    def to_unitl2_norm(self, fts):
        if type(fts) == dict or type(fts) == defaultdict:
            norm = self.l2_norm(fts)
            return self.update_dict(fts, norm)
        elif type(fts) == list:
            a = []
            for item in fts:
                a.append(self.to_unitl2_norm(item))
            return a
        else:
            raise Exception("Unexpected type: " + str(type(fts)))

    def proj(self, weights):
        l2_n = self.l2_norm(weights)
        denom = max(1, l2_n)
        return self.update_dict(weights, denom)

    def update_dict(self, dct, denom):
        u = defaultdict(float)
        for k, v in dct.items():
            u[k] = v / denom
        return u

    def clone(self):
        clone = ALMA(self.features)
        clone.weights.update(self.weights)
        clone.k = self.k
        return clone

    def rank(self, features_array):
        normed_array = self.to_unitl2_norm(features_array)
        '''Dot-product the features and current weights and return the best label.'''
        scores2index = {}
        for i, feats in enumerate(normed_array):
            scores2index[i] = self.decision_function(feats)
        # return a ranking of the scores, by best to worse
        return [ix for ix, score in sorted(scores2index.items(), key=lambda tpl: -tpl[-1])]

    def decision_function(self, features):
        '''Dot-product the features and current weights and return the score.'''
        score = 0.0
        for feat, value in features.items():
            if value == 0 or feat not in self.features:
                continue
            score += self.weights[feat] * value
        return score

    def weight_product(self, features):
        '''Dot-product the features and current weights and return the score.'''
        prod = defaultdict(float)
        for feat, value in features.items():
            if value == 0:
                continue
            prod[feat] = self.weights[feat] * value
        return prod

    def add_dicts(self, d1, d2):
        for k,v in d2.items():
            d1[k] += v
        return d1

    def train(self, best_feats, other_feats_array):

        feats_array = [best_feats] + list(other_feats_array)
        ixs = self.rank(feats_array)

        # go thru up to |max_update_items| items ranked above the best, and update the weights
        best_ix = ixs[0]
        if best_ix == 0:
            return

        best_fts_prod = self.weight_product(self.to_unitl2_norm(best_feats))
        num_other_feats = len(other_feats_array)
        assert num_other_feats > 0
        # total up other features vectors
        other_feats_product = defaultdict(float)
        for fts in other_feats_array:
            product = self.weight_product(self.to_unitl2_norm(fts))
            self.add_dicts(other_feats_product, product)

        delta = dict()
        # normalize by the number of other features
        for feat in self.features:
            delta[feat] = best_fts_prod[feat] - (other_feats_product[feat] / num_other_feats) # need to normalize the other feats value

        proj_delta = self.proj(delta)
        new_weights = defaultdict(float)
        for ft in self.features:
            new_weights[ft] = self.weights[ft] + (self.C * self.k**-0.5 * proj_delta[ft])
        new_weights = self.proj(new_weights)
        self.weights = new_weights

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None


# p = ALMA(features={"a","b","c"})
# best = defaultdict(float)
# best.update({ "a": 1, "b": 2})
#
# rest = defaultdict(float)
# rest.update({"a": -1, "b": 3, "c": 4})
#
# p.train(best, [rest])