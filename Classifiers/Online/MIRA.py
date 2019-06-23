from collections import defaultdict
import numpy as np

from Classifiers.Online.structured_perceptron import StructuredPerceptron

class MIRA(StructuredPerceptron):
    ''' MIRA Algorithm for multi-class classification as detailed in p 569 of
        http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
    '''

    def __init__(self, C=0.01, max_update_items=1, pa_type=1, initial_weight=1):

        super(MIRA, self).__init__(max_update_items=max_update_items, initial_weight=initial_weight, learning_rate=1)

        self.C = C
        self.pa_type = pa_type
        assert self.pa_type in {0, 1, 2}  # PA I or PA II
        self.train_feats = set()

    def clone(self):
        cloney = MIRA(C=self.C, max_update_items=self.max_update_items, pa_type=self.pa_type)
        cloney.weights.update(self.weights)
        cloney._totals.update(self._totals)
        cloney._tstamps.update(self._tstamps)
        cloney.i = self.i
        cloney.train_feats.update(self.train_feats)
        return cloney

    def train(self, best_feats, other_feats_array):
        if len(other_feats_array) == 0:
            return

        best_feats_score = self.decision_function(best_feats, existence_check=False)
        scores = [self.decision_function(feats, existence_check=False) for feats in other_feats_array]
        ixs = np.argsort(scores)[::-1]

        # go thru up to |max_update_items| items ranked above the best, and update the weights
        for rank, ix in enumerate(ixs):
            if rank >= self.max_update_items:
                break
            feats_score = scores[ix]
            diff = best_feats_score - feats_score
            hinge_loss = 0 if diff >= 1 else 1 - diff
            if hinge_loss > 0:
                self.update(loss=hinge_loss, best_feats=best_feats, highest_ranked_feats=other_feats_array[ix])

    def update(self, loss, best_feats, highest_ranked_feats):
        self.i += 1
        feats_union = set(best_feats.keys()).union(highest_ranked_feats.keys())
        sum_sq_diff = 0
        for ft in feats_union:
            sum_sq_diff += (best_feats[ft] - highest_ranked_feats[ft]) ** 2
        l2_norm_of_diffs = (sum_sq_diff ** 0.5)

        if sum_sq_diff == 0 and self.pa_type in {0, 1}:
            tau = self.C
        elif self.pa_type == 0:
            tau = loss / l2_norm_of_diffs
        elif self.pa_type == 1:
            tau = min(self.C, loss / l2_norm_of_diffs)
        else: #self.pa_type == 2:
            tau = loss / (l2_norm_of_diffs + 1 / (2 * self.C))

        for feat, weight in self.weights.items():
            self.train_feats.add(feat)
            val = tau * (best_feats[feat] - highest_ranked_feats[feat])
            self.__upd_feat__(feat, val)
        return None

class CostSensitiveMIRA(MIRA):

    def __init__(self, C=0.01, max_update_items=1, pa_type=1, loss_type="pb", initial_weight=1):

        assert loss_type in {"pb","ml"}, "Unrecognized loss type: {loss_type}".format(loss_type=loss_type)

        self.loss_type = loss_type
        super(CostSensitiveMIRA, self).__init__(
            C=C, max_update_items=max_update_items, pa_type=pa_type, initial_weight=initial_weight)

    def clone(self):
        cloney = CostSensitiveMIRA(
            C=self.C, max_update_items=self.max_update_items, pa_type=self.pa_type, loss_type=self.loss_type)
        cloney.weights.update(self.weights)
        cloney._totals.update(self._totals)
        cloney._tstamps.update(self._tstamps)
        cloney.i = self.i
        cloney.train_feats.update(self.train_feats)
        return cloney

    def train(self, best_feats, other_feats_array, other_costs_array):

        if len(other_feats_array) == 0:
            return

        best_feats_score = self.decision_function(best_feats, existence_check=False)
        other_feat_scores = [self.decision_function(feats, existence_check=False) for feats in other_feats_array]
        cs_losses = np.asarray(other_feat_scores) - best_feats_score + (np.asarray(other_costs_array) ** 0.5)

        if self.loss_type == "ml":
            ixs = np.argsort(cs_losses)[::-1]
        else:
            ixs = np.argsort(other_feat_scores)[::-1]

        # go thru up to |max_update_items| items ranked above the best, and update the weights
        for rank, ix in enumerate(ixs):
            if rank >= self.max_update_items:
                break

            cost_sensitive_loss = cs_losses[ix]
            if cost_sensitive_loss > 0:
                self.update(loss=cost_sensitive_loss, best_feats=best_feats, highest_ranked_feats=other_feats_array[ix])

if __name__ == "__main__":

    p = CostSensitiveMIRA(loss_type="ml")
    best = defaultdict(float)
    best.update({ "a": 1, "b": 2})

    rest1 = defaultdict(float)
    rest1.update({"a": -1, "b": 3, "c": 4})

    rest2 = defaultdict(float)
    rest2.update({"a": 2, "b": 1, "c": 2})

    p.train(best, [rest1, rest2], [1, 5])
    pass
