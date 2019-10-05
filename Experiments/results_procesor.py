import BrattEssay
import Rpfa

__author__ = 'simon.hughes'

import pymongo
from Rpfa import mean_rpfa, weighted_mean_rpfa, rpfa, micro_rpfa
from Metrics import compute_tp_fp_fn, rpf1a_from_tp_fp_tn_fn
from collections import defaultdict
import datetime
import dill
import pandas

__MACRO_F1__ = "MACRO_F1"
__MICRO_F1__ = "MICRO_F1"

def is_a_regular_code(code):
    return (code[0].isdigit() or code[0].lower() == 'p') \
            and "->" not in code and ":" not in code

def metrics_to_df(metrics):
    """
    Given a metrics Dictionary, returns a Pandas DataFrame
    :param metrics:
    :return:
    """
    rows = []
    for k,val in metrics.items():
        if type(val) == Rpfa.rpfa:
            d = dict(val.__dict__) # convert obj to dict
        elif type(val) == dict:
            d = dict(val)
        else:
            d = dict()
        d["code"] = k
        rows.append(d)
    return pandas.DataFrame(rows)

# Check mongo is running
def is_mongo_runnning():
    import pymongo
    client = pymongo.MongoClient(serverSelectionTimeoutMS=100, host="127.0.0.1")
    db = client.metrics_codes
    coll = db.get_collection("CB_TAGGING_TD_AVG_PERCEPTRON_MOST_COMMON_TAG")
    l = list(coll.find({}))


class ResultsProcessor(object):

    def __init__(self, dbname = None, fltr = None):
        # Run this here so any code referencing this module will blow up when mongo is not running
        # SH: - confirmed this did work as intended
        is_mongo_runnning()
        
        if dbname is None:
            dbname = "metrics"
            print("WARNING - No db name specified - should be either 'metrics_causal' or 'metrics'. Defaulting to 'metrics' ")

        if not fltr:
            fltr = lambda k: k[0].isdigit()

        self.fltr = fltr
        self.client = pymongo.MongoClient(serverSelectionTimeoutMS=100, host="127.0.0.1")
        self.db = self.client[dbname]
        self.dbname = dbname

    def __add_meta_data_(self, db_row, experiment_args):
        db_row["parameters"] = experiment_args
        db_row["asof"] = datetime.datetime.now()

    @staticmethod
    def compute_metrics(ys_by_tag, predictions_by_tag):
        """ Compute metrics for all predicted codes """
        metrics_by_tag = dict()
        for tag, pred_ys in predictions_by_tag.items():
            try:
                ys = ys_by_tag[tag]
                if len(ys) == 0:
                    continue

                tp, fp, fn = compute_tp_fp_fn(ys, pred_ys)
                tn = len(ys) - (tp + fp + fn)
                r, p, f1, acc = rpf1a_from_tp_fp_tn_fn(tp, fp, tn, fn)
                metric = rpfa(r, p, f1, acc,
                              nc=len([1 for y in ys if y > 0.0]), data_points=len(ys),
                              tp=tp, fp=fp, tn=tn, fn=fn)
                metrics_by_tag[tag] = metric
            except Exception as e:
                print("Exception processing tag: %s" % str(tag))
                raise e
        return metrics_by_tag

    @staticmethod
    def compute_mean_metrics(ys_by_tag, predictions_by_tag, fltr=is_a_regular_code):
        metrics = ResultsProcessor.compute_metrics(ys_by_tag, predictions_by_tag)
        return ResultsProcessor.add_mean_metrics(metrics, fltr)

    @staticmethod
    def __get_label_(tag, expected_tag_set):
        if tag in expected_tag_set:
            return 1
        else:
            return 0

    @staticmethod
    def get_wd_level_lbs(essays, expected_tags):
        expected_tags = set(expected_tags)
        ysbycode = defaultdict(list)
        for e in essays:
            for sent in e.sentences:
                for wd, tag_set in sent:
                    for etag in expected_tags:
                        ysbycode[etag].append(ResultsProcessor.__get_label_(etag, tag_set))
        return ysbycode

    @staticmethod
    def get_wd_level_preds(essays, expected_tags):
        expected_tags = set(expected_tags)
        ysbycode = defaultdict(list)
        for e in essays:
            for sentix in range(len(e.sentences)):
                p_ccodes = e.pred_tagged_sentences[sentix]
                for wordix in range(len(p_ccodes)):
                    ptag_set = {p_ccodes[wordix]}
                    assert len(ptag_set) >= 1, "No tags found"
                    for exp_tag in expected_tags:
                        ysbycode[exp_tag].append(ResultsProcessor.__get_label_(exp_tag, ptag_set))
        return ysbycode

    @staticmethod
    def compute_mean_metrics_from_tagged_essays(tagged_esssays):
        """
        Given a set of already tagged essays, computes the mean metrics using the BrattEssay.pred_tagged_sentences list
        :param tagged_esssays:
        :param expected_tags:
        :return:
        """

        # this give flexibility to look at anaphora, cc or both
        act_ys_bycode  = ResultsProcessor.get_wd_level_lbs(tagged_esssays, expected_tags)
        pred_ys_bycode = ResultsProcessor.get_wd_level_preds(tagged_esssays, expected_tags)

        assert len(act_ys_bycode.keys()) == len(pred_ys_bycode.keys()) == len(expected_tags), "Miss-matched codes"
        first_tag = list(expected_tags)[0]
        last_tag  = list(expected_tags)[-1]
        assert len(act_ys_bycode[first_tag]) == len(pred_ys_bycode[first_tag]), "Different numbers of words"
        assert len(act_ys_bycode[last_tag]) == len(pred_ys_bycode[last_tag]),   "Different numbers of words"

        return ResultsProcessor.compute_mean_metrics(act_ys_bycode, pred_ys_bycode)

    @staticmethod
    def compute_metrics_from_tagged_essays(
            tagged_esssays, expected_tags):
        """
        Given a set of already tagged essays, computes the regular metrics using the BrattEssay.pred_tagged_sentences list
        :param tagged_esssays:
        :param expected_tags:
        :return:
        """

        # this give flexibility to look at anaphora, cc or both
        act_ys_bycode  = ResultsProcessor.get_wd_level_lbs(tagged_esssays,   expected_tags)
        pred_ys_bycode = ResultsProcessor.get_wd_level_preds(tagged_esssays, expected_tags)

        assert len(act_ys_bycode.keys()) == len(pred_ys_bycode.keys()) == len(expected_tags), "Miss-matched codes"
        first_tag = list(expected_tags)[0]
        last_tag  = list(expected_tags)[-1]
        assert len(act_ys_bycode[first_tag]) == len(pred_ys_bycode[first_tag]), "Different numbers of words"
        assert len(act_ys_bycode[last_tag]) == len(pred_ys_bycode[last_tag]),   "Different numbers of words"

        return ResultsProcessor.compute_metrics(act_ys_bycode, pred_ys_bycode)

    @staticmethod
    def add_mean_metrics(dict_mean_metrics, fltr=is_a_regular_code):

        mean_metrics = dict(dict_mean_metrics.items())
        code_metrics = []
        # Filters to concept codes by default
        for tag, metric in mean_metrics.items():
            if fltr(tag):
                code_metrics.append(metric)

        """ All Tags """
        mean_metric = mean_rpfa(mean_metrics.values())
        weighted_mean_metric = weighted_mean_rpfa(mean_metrics.values())
        micro_f1_metric = micro_rpfa(mean_metrics.values())

        """ Concept Codes """
        mean_metric_codes = mean_rpfa(code_metrics)
        weighted_mean_metric_codes = weighted_mean_rpfa(code_metrics)

        mean_metrics["MEAN"] = mean_metric
        mean_metrics["WEIGHTED_MEAN"] = weighted_mean_metric

        """ The default behavior is to assume codes starting with a digit are concept codes """
        mean_metrics["MEAN_CONCEPT_CODES"] = mean_metric_codes
        mean_metrics[
            "WEIGHTED_MEAN_CONCEPT_CODES"] = weighted_mean_metric_codes  # convert values to dicts from rpfa objects for mongodb

        """ Micro and Macro F1 """
        mean_metrics[__MICRO_F1__] = micro_f1_metric
        macro_f1 = ResultsProcessor.f1(mean_metric_codes.recall, mean_metric_codes.precision)
        return dict(list(map(lambda tpl: (tpl[0], tpl[1].__dict__), mean_metrics.items())) + [(__MACRO_F1__, macro_f1)])

    @staticmethod
    def f1(r, p):
        r, p = float(r), float(p)
        denom = (r + p)
        if denom == 0:
            return 0.0
        return (2.0 * r * p) / denom

    def persist_predictions(self, dbcollection, predictions_by_tag, ys_by_tag):
        # Persist predictions (too large to store in mongo
        folder = "/Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Predictions/"
        n = datetime.datetime.now()
        ts = "{y}-{m}-{d}_{h}-{minut}".format(y=n.year, m=n.month, d=n.day, h=n.hour, minut=n.time().minute)
        suffix_ts = "_" + ts + ".dill"
        suffix = ".dill"

        f_ys = folder + dbcollection + "_YS_" + suffix
        f_ys_ts = folder + dbcollection + "_YS_" + suffix_ts
        with open(f_ys, "wb+") as f:
            dill.dump(dict(ys_by_tag.items()), f)
        with open(f_ys_ts, "wb+") as f:
            dill.dump(dict(ys_by_tag.items()), f)

        f_preds    = folder + dbcollection + "_PREDS_" + suffix
        f_preds_ts = folder + dbcollection + "_PREDS_" + suffix_ts
        with open(f_preds, "wb+") as f:
            dill.dump(dict(predictions_by_tag.items()), f)
        with open(f_preds_ts, "wb+") as f:
            dill.dump(dict(predictions_by_tag.items()), f)

    def persist_results(self, dbcollection, ys_by_tag, predictions_by_tag, experiment_args, algorithm, **kwargs):

        self.persist_predictions(dbcollection, predictions_by_tag, ys_by_tag)

        experiment_args["num_tags"] = len(ys_by_tag.keys())
        # Compute Mean metrics over all folds
        metrics_by_code        = ResultsProcessor.compute_metrics(ys_by_tag, predictions_by_tag)
        mean_td_metrics_by_tag = ResultsProcessor.add_mean_metrics(metrics_by_code, self.fltr)

        db_row = dict(mean_td_metrics_by_tag.items())
        db_row["algorithm"] = algorithm
        # merge in additional values
        for key, val in kwargs.items():
            db_row[key] = val
        self.__add_meta_data_(db_row, experiment_args)
        return self.db[dbcollection].insert(db_row)

    @staticmethod
    def __metrics_to_str__(pad_str, tag, td_rpfa, vd_rpfa):
        s_metrics = ""
        s_metrics += "\nTAG:       " + ResultsProcessor.pad_str(tag)
        s_metrics += "\nf1:        " + ResultsProcessor.pad_str(td_rpfa["f1_score"]) + pad_str(vd_rpfa["f1_score"])
        s_metrics += "\nrecall:    " + ResultsProcessor.pad_str(td_rpfa["recall"]) + pad_str(vd_rpfa["recall"])
        s_metrics += "\nprecision: " + ResultsProcessor.pad_str(td_rpfa["precision"]) + pad_str(vd_rpfa["precision"])
        s_metrics += "\naccuracy:  " + ResultsProcessor.pad_str(td_rpfa["accuracy"]) + pad_str(vd_rpfa["accuracy"])
        s_metrics += "\nsentences: " + ResultsProcessor.pad_str("") + pad_str(vd_rpfa["num_codes"])
        s_metrics += "\n"
        return s_metrics

    @staticmethod
    def pad_str(val):
        return str(val).ljust(20) + "  "

    @staticmethod
    def __sort_key_(code):
        if code.isdigit():
            return (int(code), len(code), code)
        return (9999999, 9999999, not code[0].isdigit(), not code.startswith("_"), code[-1].isupper(), code)


    @staticmethod
    def metrics_to_string(td_metrics, vd_metrics, header):
        s_metrics = ""
        s_metrics += header + "\n"

        std_metrics = sorted(td_metrics.items(), key=lambda tpl: ResultsProcessor.__sort_key_(tpl[0]))
        svd_metrics = sorted(vd_metrics.items(), key=lambda tpl: ResultsProcessor.__sort_key_(tpl[0]))

        micro_f1 = ""
        for ((tag, td_rpfa), (_, vd_rpfa)) in zip(std_metrics, svd_metrics):
            if type(td_rpfa) == dict and "f1_score" in td_rpfa:
                s_metric = ResultsProcessor.__metrics_to_str__(ResultsProcessor.pad_str, tag, td_rpfa, vd_rpfa)
                if tag == __MICRO_F1__:
                    micro_f1 = s_metric
                else:
                    s_metrics += s_metric

        # Make sure this is at the end
        s_metrics += micro_f1
        s_metrics += "\nMacro F1:  " + ResultsProcessor.pad_str(td_metrics[__MACRO_F1__]) + ResultsProcessor.pad_str(vd_metrics[__MACRO_F1__])
        s_metrics += "\n"
        return s_metrics

    def results_to_string(self, td_objectid, td_collection, vd_objectid, vd_collection, header):

        td_metrics = self.db[td_collection].find_one({"_id": td_objectid})
        vd_metrics = self.db[vd_collection].find_one({"_id": vd_objectid})

        return ResultsProcessor.metrics_to_string(td_metrics, vd_metrics, header)

    def get_metric(self, collection, objectid, metric_key):

        metrics = self.db[collection].find_one({"_id": objectid})
        return metrics[metric_key]



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
    print(s)