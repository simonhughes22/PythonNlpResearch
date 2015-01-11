
from Rpfa import rpfa

def merge_metrics(src, tgt):
    for k, metric in src.items():
        tgt[k].append(metric)

def agg_metrics(src, agg_fn):
    agg = dict()
    for k, metrics in src.items():
        agg[k] = agg_fn(metrics)
    return agg

# compute mean rpfa across CV folds
def cv_mean_rpfa(metrics):
    tr, tp, tf, ta, num_codes = 0.0, 0.0, 0.0, 0.0, 0.0
    for metric in metrics:
        tr += metric.recall
        tp += metric.precision
        tf += metric.f1_score
        ta += metric.accuracy
        num_codes += metric.num_codes

    l = float(len(metrics))
    return rpfa(tr / l, tp / l, tf / l, ta / l, num_codes / l)

def cv_mean_rpfa_total_codes(metrics):
    tr, tp, tf, ta, num_codes = 0.0, 0.0, 0.0, 0.0, 0.0
    for metric in metrics:
        tr += metric.recall
        tp += metric.precision
        tf += metric.f1_score
        ta += metric.accuracy
        num_codes += metric.num_codes

    l = float(len(metrics))
    return rpfa(tr / l, tp / l, tf / l, ta / l, num_codes)