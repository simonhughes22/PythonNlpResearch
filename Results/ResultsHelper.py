
import numpy as np
from Metrics import *
from DictionaryHelper import tally_items
from ListHelper import filter_list_by_index

def rfp(expected, actual):

    r = recall    (expected, actual)
    p = precision (expected, actual)
    f1 = f_beta   (expected, actual, beta = 1.0)
    acc = accuracy(expected, actual)

    num_positive = len([l for l in expected if l == 1])
    results = ""
    
    results += "Number of records           : " + str(len(expected)) + "\n"
    results += "Number of records for class : " + str(num_positive) + "\n"
    results += "Proportion                  : " + str(num_positive / float(len(expected))) + "\n"
    results += "\n"

    results +=  "Recall                     : " + str(r) + "\n"
    results +=  "Precision                  : " + str(p) + "\n"
    results +=  "F1 score                   : " + str(f1) + "\n"
    results +=  "ACCURACY:                  : " + str(acc * 100) + "%\n"
    results += "\n"
    return results

def dump_hits_and_misses(hits_and_misses_by_code, xs, fname):
    """ Dumps hits and misses to file, aggregating across k folds
        and sorting by frequency within the group (tp, fn, etc)

    hits_and_misses_by_code :   dict[code] of tuples (tp, fp, fn, tn)
    xs :                        list of list of tokens (tokenized_docs)
    fname :                     file to dump results to
    """

    def tally_sort_to_string(xs, indices, label):
        sorted_tally = tally_items(indices, sort=True)

        s = "\t" + label + "\n"
        for ix, cnt in sorted_tally:
            x = xs[ix]
            s += "\t\t" + str(cnt).ljust(5) + " : " + str(x) + "\n"
        s += "\n"
        return s

    codes = sorted(hits_and_misses_by_code.keys())

    with open(fname, "w+") as f:

        for code in codes:
            code_dump = ""
            # Extract indices
            tp_ix, fp_ix, fn_ix, tn_ix = hits_and_misses_by_code[code]

            uc_code = code.upper()
            code_dump += "Code: " + uc_code + "\n"

            code_dump += tally_sort_to_string(xs, tp_ix, "True  Positives [" + uc_code + "]")
            code_dump += tally_sort_to_string(xs, tn_ix, "True  Negatives [" + uc_code + "]")
            code_dump += tally_sort_to_string(xs, fp_ix, "False Positives [" + uc_code + "]")
            code_dump += tally_sort_to_string(xs, fn_ix, "False Negatives [" + uc_code + "]")

            f.write(code_dump)
    pass