from collections import defaultdict
import numpy as np

from Classifiers.Online.structured_perceptron import StructuredPerceptron

class MIRA(StructuredPerceptron):
    ''' MIRA Algorithm for multi-class classification as detailed in p 569 of
        http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
    '''

    def __init__(self, C=0.01, max_update_items=1, pa_type=1, initial_weight=1):
        self.C = C
        # Each feature gets its own weight
        # needs to be non zero otherwise first
        assert initial_weight >= 0.0
        self.weights = defaultdict(lambda : initial_weight)
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0
        # how many items do we use to update the weights?
        self.max_update_items = max_update_items
        self.pa_type = pa_type
        assert self.max_update_items >= 1, "Max update items must be at least 1"
        assert self.pa_type in {0,1,2} # PA I or PA II

        # This isn't used, so set to 1
        self.learning_rate = 1

    def clone(self):
        p = MIRA(C=self.C, max_update_items=self.max_update_items, pa_type=self.pa_type)
        p.weights.update(self.weights)
        p._totals.update(self._totals)
        p._tstamps.update(self._tstamps)
        p.i = self.i
        return p

    def train(self, best_feats, other_feats_array):
        if len(other_feats_array) == 0:
            return

        best_feats_score = self.decision_function(best_feats)
        scores = [self.decision_function(feats) for feats in other_feats_array]
        ixs = np.argsort(scores)[::-1]

        # go thru up to |max_update_items| items ranked above the best, and update the weights
        for rank, ix in enumerate(ixs):
            if rank >= self.max_update_items:
                break
            feats_score = scores[ix]
            diff = best_feats_score - feats_score
            hinge_loss = 0 if diff >= 1 else 1 - diff
            if hinge_loss > 0:
                self.update(hinge_loss=hinge_loss, best_feats=best_feats, highest_ranked_feats=other_feats_array[ix])

    def update(self, hinge_loss, best_feats, highest_ranked_feats):
        '''Update the feature weights.'''
        self.i += 1

        feats_union = set(best_feats.keys()).union(highest_ranked_feats.keys())
        sum_sq_diff = 0
        for ft in feats_union:
            sum_sq_diff += (best_feats[ft] - highest_ranked_feats[ft]) ** 2
        l2_norm_of_diffs = (sum_sq_diff ** 0.5)

        if sum_sq_diff == 0 and self.pa_type in {0,1}:
            tau = self.C
        elif self.pa_type == 0:
            tau = hinge_loss / l2_norm_of_diffs
        elif self.pa_type == 1:
            tau = min(self.C, hinge_loss / l2_norm_of_diffs)
        else:
            tau = hinge_loss / (l2_norm_of_diffs + 1/(2 * self.C))

        for feat, weight in self.weights.items():
            val = tau * (best_feats[feat] - highest_ranked_feats[feat])
            self.__upd_feat__(feat, val)
        return None

if __name__ == "__main__":

    p = MIRA()
    best = defaultdict(float)
    best.update({ "a": 1, "b": 2})

    rest = defaultdict(float)
    rest.update({"a": -1, "b": 3, "c": 4})

    p.train(best, [rest])
    pass
