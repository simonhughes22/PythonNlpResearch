__author__ = 'simon.hughes'

import pymongo
from Rpfa import mean_rpfa, weighted_mean_rpfa
from collections import defaultdict
from datetime import datetime

class ResultsProcessor(object):

    def __init__(self, fltr = None):
        if not fltr:
            fltr = lambda k: k[0].isdigit()
        self.fltr = fltr
        self.client = pymongo.MongoClient()
        self.db = self.client.metrics

    def __f1_(self, r, p):
        r, p = float(r), float(p)
        return (2.0 * r * p) / (r + p)

    def __get_mean_metrics_(self, cv_td_wd_metrics_by_tag):
        agg_metrics = defaultdict(list)
        for fold in cv_td_wd_metrics_by_tag:
            for tag, metric in fold.items():
                agg_metrics[tag].append(metric)

        dict_mean_metrics = dict()
        code_metrics = []
        for tag, lst_metrics in agg_metrics.items():
            total_codes = sum(map(lambda m: m.num_codes, lst_metrics))

            metric = mean_rpfa(lst_metrics)
            metric.num_codes = total_codes
            dict_mean_metrics[tag] = metric
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

        dict_mean_metrics["MEAN_CONCEPT_CODES"]             = mean_metric_codes
        dict_mean_metrics["WEIGHTED_MEAN_CONCEPT_CODES"]    = weighted_mean_metric_codes  # convert values to dicts from rpfa objects for mongodb

        macro_f1 = self.__f1_(mean_metric_codes.recall, mean_metric_codes.precision)
        return dict(map(lambda (k, v): (k, v.__dict__), dict_mean_metrics.items()) + [("MACRO_F1", macro_f1)])

    def __add_meta_data_(self, metrics_by_tag, experiment_args):
        metrics_by_tag["parameters"] = experiment_args
        metrics_by_tag["asof"] = datetime.now()

    def persist_results(self, collection, cv_td_wd_metrics_by_tag, experiment_args, algorithm, **kwargs):

        # Compute Mean metrics over all folds
        mean_td_metrics_by_tag = self.__get_mean_metrics_(cv_td_wd_metrics_by_tag)
        mean_td_metrics_by_tag["algorithm"] = algorithm
        for key, val in kwargs.items():
            mean_td_metrics_by_tag[key] = val
        self.__add_meta_data_(mean_td_metrics_by_tag, experiment_args)
        return self.db[collection].insert(mean_td_metrics_by_tag)

    def results_to_string(self, td_objectid, td_collection, vd_objectid, vd_collection, header):

        def pad_str(val):
            return str(val).ljust(20) + "  "

        def sort_key(code):
            if code.isdigit():
                return (int(code), len(code), code)
            return (9999999, 9999999, code.isupper(), (code))

        s_metrics = ""
        s_metrics += header + "\n"

        td_metrics = self.db[td_collection].find_one({"_id": td_objectid})
        vd_metrics = self.db[vd_collection].find_one({"_id": vd_objectid})

        std_metrics = sorted(td_metrics.items(), key=lambda (k, v): sort_key(k))
        svd_metrics = sorted(vd_metrics.items(), key=lambda (k, v): sort_key(k))

        for ((tag, td_rpfa), (_, vd_rpfa)) in zip(std_metrics, svd_metrics):

            if type(td_rpfa) == dict and "f1_score" in td_rpfa:

                s_metrics += "\nTAG:       " + pad_str(tag)
                s_metrics += "\nf1:        " + pad_str(td_rpfa["f1_score"])     + pad_str(vd_rpfa["f1_score"])
                s_metrics += "\nrecall:    " + pad_str(td_rpfa["recall"])       + pad_str(vd_rpfa["recall"])
                s_metrics += "\nprecision: " + pad_str(td_rpfa["precision"])    + pad_str(vd_rpfa["precision"])
                s_metrics += "\naccuracy:  " + pad_str(td_rpfa["accuracy"])     + pad_str(vd_rpfa["accuracy"])
                s_metrics += "\nsentences: " + pad_str("")                      + pad_str(vd_rpfa["num_codes"])
                s_metrics += "\n"
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