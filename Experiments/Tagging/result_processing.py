from metric_processing import *
from Rpfa import mean_rpfa, weighted_mean_rpfa

def print_metrics_for_codes(td_metricsByTag, vd_metricsByTag):

    def pad_str(val):
        return str(val).ljust(20) + "  "

    def sort_key(code):
        if code.isdigit():
            return (int(code), len(code), code)
        return (9999999, len(code), (code))

    s_metrics =      "\n           " + pad_str("TD") + pad_str("VD")
    for tag in sorted(td_metricsByTag.keys(), key= sort_key):
        td_rpfa = td_metricsByTag[tag]
        vd_rpfa = vd_metricsByTag[tag]
        s_metrics += "\nTAG:       " + pad_str(tag)
        s_metrics += "\nf1:        " + pad_str(td_rpfa.f1_score)    + pad_str(vd_rpfa.f1_score)
        s_metrics += "\nrecall:    " + pad_str(td_rpfa.recall)      + pad_str(vd_rpfa.recall)
        s_metrics += "\nprecision: " + pad_str(td_rpfa.precision)   + pad_str(vd_rpfa.precision)
        s_metrics += "\naccuracy:  " + pad_str(td_rpfa.accuracy)    + pad_str(vd_rpfa.accuracy)
        s_metrics += "\nsentences: " + pad_str("")                  + pad_str(vd_rpfa.num_codes)
        s_metrics += "\n"
    return s_metrics

def get_results(wd_td_all_metricsByTag, wd_vd_all_metricsByTag, sent_td_all_metricsByTag, sent_vd_all_metricsByTag,
                wd_td_wt_mean_prfa, wd_td_mean_prfa, wd_vd_wt_mean_prfa, wd_vd_mean_prfa,
                sent_td_wt_mean_prfa, sent_td_mean_prfa, sent_vd_wt_mean_prfa, sent_vd_mean_prfa,
                fn_create_wd_cls, fn_create_sent_cls):

    wd_mean_td_metrics = agg_metrics(wd_td_all_metricsByTag, cv_mean_rpfa)
    wd_mean_vd_metrics = agg_metrics(wd_vd_all_metricsByTag, cv_mean_rpfa_total_codes)

    sent_mean_td_metrics = agg_metrics(sent_td_all_metricsByTag, cv_mean_rpfa)
    sent_mean_vd_metrics = agg_metrics(sent_vd_all_metricsByTag, cv_mean_rpfa_total_codes)

    s_results = ""
    s_results += "\nTAGGING"
    s_results += print_metrics_for_codes(wd_mean_td_metrics, wd_mean_vd_metrics)
    s_results += "\n\nSENTENCE"
    s_results += print_metrics_for_codes(sent_mean_td_metrics, sent_mean_vd_metrics)
    s_results += str(fn_create_wd_cls())
    # print macro measures
    s_results += "\n\n\nTAGGING"
    s_results += "\n\nTraining   Performance"
    s_results += "\nWeighted:" + str(cv_mean_rpfa(wd_td_wt_mean_prfa))
    s_results += "\nMean    :" + str(cv_mean_rpfa(wd_td_mean_prfa))
    s_results += "\n\nValidation Performance"
    s_results += "\nWeighted:" + str(cv_mean_rpfa_total_codes(wd_vd_wt_mean_prfa))
    s_results += "\nMean    :" + str(cv_mean_rpfa_total_codes(wd_vd_mean_prfa))
    s_results += "\n\n\n"
    s_results += str(fn_create_sent_cls())
    # print macro measures
    s_results += "\n\n\nSENTENCE"
    s_results += "\n\nTraining   Performance"
    s_results += "\nWeighted:" + str(cv_mean_rpfa(sent_td_wt_mean_prfa))
    s_results += "\nMean    :" + str(cv_mean_rpfa(sent_td_mean_prfa))
    s_results += "\n\nValidation Performance"
    s_results += "\nWeighted:" + str(cv_mean_rpfa_total_codes(sent_vd_wt_mean_prfa))
    s_results += "\nMean    :" + str(cv_mean_rpfa_total_codes(sent_vd_mean_prfa))
    return s_results