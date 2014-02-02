__author__ = 'simon.hughes'

class rpfa(object):
    def __repr__(self):
        return self.to_str(True)

    def __init__(self, r, p, f, a, nc=-1):
        self.recall = r
        self.precision = p
        self.f1_score = f
        self.accuracy = a
        self.num_codes = nc

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


def mean_rpfa( metrics):
    tr, tp, tf, ta = 0.0, 0.0, 0.0, 0.0
    for metric in metrics:
        tr += metric.recall
        tp += metric.precision
        tf += metric.f1_score
        ta += metric.accuracy

    l = float(len(metrics))
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

    return rpfa(tr / total_codes, tp / total_codes, tf / total_codes, ta / total_codes, total_codes)
