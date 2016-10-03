__author__ = 'simon.hughes'

from Metrics import rpf1a, rpf1a_from_tp_fp_tn_fn

class rpfa(object):
    def __repr__(self):
        return self.to_str(True)

    def __init__(self, r, p, f, a, nc=-1, data_points=None, tp=None, tn=None, fp=None, fn=None):
        self.recall = r
        self.precision = p
        self.f1_score = f
        self.accuracy = a
        self.num_codes = nc

        if data_points:
            self.data_points = data_points
        if tp:
            self.tp = tp
        if tn:
            self.tn = tn
        if fp:
            self.fp = fp
        if fn:
            self.fn = fn

    def to_str(self, print_count=False):
        fmt = "Recall: {0:.4f}, Precision: {1:.4f}, F1: {2:.4f}, Accuracy: {3:.4f}"
        if print_count:
            fmt += ", Codes: {4}"
        return fmt.format(
            self.recall,
            self.precision,
            self.f1_score,
            self.accuracy,
            str(int(self.num_codes)).rjust(5))


def mean_rpfa(metrics):
    tr, tp, tf, ta = 0.0, 0.0, 0.0, 0.0
    for metric in metrics:
        tr += metric.recall
        tp += metric.precision
        tf += metric.f1_score
        ta += metric.accuracy

    l = float(len(metrics))
    #prevent divide by 0!
    if l == 0.0:
        return rpfa(0,0,0,0,0)
    return rpfa(tr / l, tp / l, tf / l, ta / l, l)


def weighted_mean_rpfa(metrics):
    tr, tp, tf, ta = 0.0, 0.0, 0.0, 0.0

    total_codes = 0.0

    for metric in metrics:
        tr += metric.recall * metric.num_codes
        tp += metric.precision * metric.num_codes
        tf += metric.f1_score * metric.num_codes
        ta += metric.accuracy * metric.num_codes
        total_codes += metric.num_codes

    if total_codes <= 0.0:
        return rpfa(0.0,0.0,0.0,0.0)
    return rpfa(tr / total_codes, tp / total_codes, tf / total_codes, ta / total_codes, total_codes)

def micro_rpfa(metrics):
    sum_tp, sum_fp, sum_tn, sum_fn = 0.0, 0.0, 0.0, 0.0
    total_codes = 0.0
    data_points = 0.0

    for metric in metrics:
        # Back into the micro F1 score from the existing calcs
        sum_tp += metric.tp
        sum_fp += metric.fp
        sum_tn += metric.tn
        sum_fn += metric.fn

        data_points += metric.data_points
        total_codes += metric.num_codes

    if total_codes <= 0.0:
        return rpfa(0.0,0.0,0.0,0.0)

    r,p,f1,a = rpf1a_from_tp_fp_tn_fn(sum_tp, sum_fp, sum_tn, sum_fn)
    return rpfa(r, p, f1, a, nc=total_codes, data_points=data_points)
