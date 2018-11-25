import pandas as pd
from collections import defaultdict
from results_procesor import ResultsProcessor
from results_procesor import metrics_to_df

# Modify this function from the Resultsprocessor so that it works with Set[str] of predicted tags 
# as well as scalar strings
def get_wd_level_preds(essays, expected_tags):
    expected_tags = set(expected_tags)
    ysbycode = defaultdict(list)
    for e in essays:
        for sentix in range(len(e.sentences)):
            p_ccodes = e.pred_tagged_sentences[sentix]
            for wordix in range(len(p_ccodes)):
                tags = p_ccodes[wordix]
                if type(tags) == str:
                    ptag_set = {tags}
                elif type(tags) in (set,list):
                    ptag_set = set(tags)   
                else:
                    raise Exception("Unrecognized tag type")
                for exp_tag in expected_tags:
                    ysbycode[exp_tag].append(ResultsProcessor._ResultsProcessor__get_label_(exp_tag, ptag_set))
    return ysbycode

def get_metrics_raw(essays, expected_tags, micro_only=False):
    act_ys_bycode  = ResultsProcessor.get_wd_level_lbs(essays,  expected_tags=expected_tags)
    pred_ys_bycode = get_wd_level_preds(essays, expected_tags=expected_tags)
    mean_metrics = ResultsProcessor.compute_mean_metrics(act_ys_bycode, pred_ys_bycode)
    return mean_metrics