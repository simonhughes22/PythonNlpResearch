__author__ = 'simon.hughes'

import pymongo
from Rpfa import mean_rpfa, weighted_mean_rpfa, rpfa
from Metrics import rpf1a
from collections import defaultdict
from datetime import datetime

__MACRO_F1__ = "MACRO_F1"

def compute_metrics(ys_by_tag, predictions_by_tag):
    """ Compute metrics for all predicted codes """
    metrics_by_tag = dict()
    for tag, pred_ys in predictions_by_tag.items():
        try:
            ys = ys_by_tag[tag]
            if len(ys) == 0:
                continue
            r, p, f1, acc = rpf1a(ys, pred_ys)
            metric = rpfa(r, p, f1, acc, nc=len([1 for y in ys if y > 0.0]))
            metrics_by_tag[tag] = metric
        except Exception as e:
            print("Exception processing tag: %s" % str(tag))
            raise e
    return metrics_by_tag

class ResultsProcessor(object):

    def __init__(self, fltr = None):
        if not fltr:
            fltr = lambda k: k[0].isdigit()
        self.fltr = fltr
        self.client = pymongo.MongoClient()
        self.db = self.client.metrics

    def __f1_(self, r, p):
        r, p = float(r), float(p)
        denom = (r + p)
        if denom == 0:
            return 0.0
        return (2.0 * r * p) / denom

    def __get_mean_metrics_(self, dict_mean_metrics):

        code_metrics = []
        for tag, metric in dict_mean_metrics.items():
            if self.fltr(tag):
                code_metrics.append(metric)

        """ All Tags """
        mean_metric = mean_rpfa(dict_mean_metrics.values())
        weighted_mean_metric = weighted_mean_rpfa(dict_mean_metrics.values())

        """ Concept Codes """
        mean_metric_codes = mean_rpfa(code_metrics)
        weighted_mean_metric_codes = weighted_mean_rpfa(code_metrics)

        dict_mean_metrics["MEAN"]                           = mean_metric
        dict_mean_metrics["WEIGHTED_MEAN"]                  = weighted_mean_metric

        """ The default behavior is to assume codes starting with a digit are concept codes """
        dict_mean_metrics["MEAN_CONCEPT_CODES"]             = mean_metric_codes
        dict_mean_metrics["WEIGHTED_MEAN_CONCEPT_CODES"]    = weighted_mean_metric_codes  # convert values to dicts from rpfa objects for mongodb

        macro_f1 = self.__f1_(mean_metric_codes.recall, mean_metric_codes.precision)
        return dict(map(lambda (k, v): (k, v.__dict__), dict_mean_metrics.items()) + [(__MACRO_F1__, macro_f1)])

    def __add_meta_data_(self, db_row, experiment_args):
        db_row["parameters"] = experiment_args
        db_row["asof"] = datetime.now()

    def persist_results(self, dbcollection, ys_by_tag, predictions_by_tag, experiment_args, algorithm, **kwargs):

        experiment_args["num_tags"] = len(ys_by_tag.keys())
        # Compute Mean metrics over all folds
        metrics_by_code = compute_metrics(ys_by_tag, predictions_by_tag)
        mean_td_metrics_by_tag = self.__get_mean_metrics_(metrics_by_code)

        db_row = dict(mean_td_metrics_by_tag.items())
        db_row["algorithm"] = algorithm
        # merge in additional values
        for key, val in kwargs.items():
            db_row[key] = val
        self.__add_meta_data_(db_row, experiment_args)
        return self.db[dbcollection].insert(db_row)

    def __metrics_to_str__(self, pad_str, tag, td_rpfa, vd_rpfa):
        s_metrics = ""
        s_metrics += "\nTAG:       " + pad_str(tag)
        s_metrics += "\nf1:        " + pad_str(td_rpfa["f1_score"]) + pad_str(vd_rpfa["f1_score"])
        s_metrics += "\nrecall:    " + pad_str(td_rpfa["recall"]) + pad_str(vd_rpfa["recall"])
        s_metrics += "\nprecision: " + pad_str(td_rpfa["precision"]) + pad_str(vd_rpfa["precision"])
        s_metrics += "\naccuracy:  " + pad_str(td_rpfa["accuracy"]) + pad_str(vd_rpfa["accuracy"])
        s_metrics += "\nsentences: " + pad_str("") + pad_str(vd_rpfa["num_codes"])
        s_metrics += "\n"
        return s_metrics

    def results_to_string(self, td_objectid, td_collection, vd_objectid, vd_collection, header):

        def pad_str(val):
            return str(val).ljust(20) + "  "

        def sort_key(code):
            if code.isdigit():
                return (int(code), len(code), code)
            return (9999999, 9999999, not code[0].isdigit(), not code.startswith("_"), code[-1].isupper(), code)

        s_metrics = ""
        s_metrics += header + "\n"

        td_metrics = self.db[td_collection].find_one({"_id": td_objectid})
        vd_metrics = self.db[vd_collection].find_one({"_id": vd_objectid})

        std_metrics = sorted(td_metrics.items(), key=lambda (k, v): sort_key(k))
        svd_metrics = sorted(vd_metrics.items(), key=lambda (k, v): sort_key(k))

        for ((tag, td_rpfa), (_, vd_rpfa)) in zip(std_metrics, svd_metrics):
            if type(td_rpfa) == dict and "f1_score" in td_rpfa:
                s_metrics += self.__metrics_to_str__(pad_str, tag, td_rpfa, vd_rpfa)

        s_metrics         += "\nMacro F1:  " + pad_str(td_metrics[__MACRO_F1__]) + pad_str(vd_metrics[__MACRO_F1__])
        s_metrics         += "\n"
        return s_metrics

if __name__ == "__main__":

    import pymongo

    client = pymongo.MongoClient()
    db = client.metrics

    td_collection = "CB_TAGGING_TD"
    vd_collection = "CB_TAGGING_VD"

    td_m = db[td_collection].find_one()
    vd_m = db[vd_collection].find_one()

    processor = ResultsProcessor()
    s = processor.results_to_string(td_m["_id"], td_collection, vd_m["_id"], vd_collection, "TAGGING")
    print s