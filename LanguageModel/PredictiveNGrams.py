from collections import defaultdict
from NgramGenerator import compute_ngrams
from SkipGramGenerator import compute_skip_grams
from IterableFP import *

""" Smooth counts """

def __tally_initial_counts__(labels_for_class, tokenized_docs):
    # Tally initial class counts and token doc freq

    # How many labels per class
    class_counts = defaultdict(float)
    # For optimization. If single word doc freq < threshold,
    # then ngrams > 1 will not be over threshold either
    token_doc_freq = defaultdict(float)
    for doc, lbl in zip(tokenized_docs, labels_for_class):
        class_counts[lbl] += 1.0
        for w in set(doc):
            token_doc_freq[w] += 1.0

    return class_counts, token_doc_freq

def get_predictive_ngrams(tokenized_docs, labels_for_class, ngram_gen_fn, ngram_size_limit, min_doc_freq = 5):

    """
    tokenized_docs :    list of list of ordered tokens (one list per doc)
    labels_for_class :  labels for the class to be classified
    ngram_size_limit :  max size ngram to compute
    min_doc_freq :      minimum doc freq to consider
    """

    class_counts, token_doc_freq = __tally_initial_counts__(labels_for_class, tokenized_docs)
    if min(class_counts.values()) == 0:
        raise Exception("One or more classes has not instances in the dataset")

    wds_to_consider = set( wd for wd, freq in token_doc_freq.items() if freq >= min_doc_freq )

    def is_positive(lbl):
        return lbl > 0

    total_freq = defaultdict(lambda: 0.0)
    positive_tally = defaultdict(lambda: 0.0)
    negative_tally = defaultdict(lambda: 0.0)

    str2ngram = dict()
    below_freq = set()
    for i, (doc, lbl) in enumerate(zip(tokenized_docs, labels_for_class)):

        #print i
        ngrams = ngram_gen_fn(doc, ngram_size_limit)
        if ngrams is None: # length one skip grams return None
            continue
        unique_ngrams_for_doc = set()
        for ngram in ngrams:
            str_ngram = str(ngram)
            if str_ngram in unique_ngrams_for_doc or str_ngram in below_freq:
                continue

            # if one of the constituents does not meet the min freq
            # then ignore
            if any(wd not in wds_to_consider for wd in ngram):
                below_freq.add(str_ngram)
                continue

            unique_ngrams_for_doc.add(str_ngram)
            str2ngram[str_ngram] = ngram

            total_freq[str_ngram] += 1
            # Conditional tally
            if is_positive(lbl):
                positive_tally[str_ngram] += 1
            else:
                negative_tally[str_ngram] += 1

    conditional_positive_ratios = dict()
    conditional_negative_ratios = dict()

    pos_label = max(class_counts.keys())
    neg_label = min(class_counts.keys())

    # To prevent div! 0 errors, estimate the minimal conditionals to be 0.5 counts,
    # so 0.5 the observed minimum freq
    min_pos_cond_prob = 0.5 / class_counts[pos_label]
    min_negative_cond_prob = 0.5 / class_counts[neg_label]

    for str_ngram, prior_count in total_freq.items():
        if total_freq[str_ngram] < min_doc_freq:
            continue

        positive_conditional_prob  = positive_tally[str_ngram] / class_counts[pos_label]
        if positive_conditional_prob == 0.0:
            positive_conditional_prob = min_pos_cond_prob

        negative_conditional_prob  = negative_tally[str_ngram] / class_counts[neg_label]
        if negative_conditional_prob == 0.0:
            negative_conditional_prob = min_negative_cond_prob

        conditional_positive_ratios[str_ngram] = positive_conditional_prob / negative_conditional_prob * 1.0
        conditional_negative_ratios[str_ngram] = negative_conditional_prob / positive_conditional_prob * 1.0

    """ sort first by size """
    #sratio = sorted(conditional_ratios.items(), key = lambda item: len(item[0]))
    sorted_positive_ratios = sorted(conditional_positive_ratios.items(), key = lambda item: -item[1])
    sorted_negative_ratios = sorted(conditional_negative_ratios.items(), key = lambda item: -item[1])

    return (sorted_positive_ratios, sorted_negative_ratios, positive_tally, negative_tally, total_freq,  str2ngram)

def print_ratios(sratios, prior_tally, top_n = 100):
    print "Ngram".ljust(80), "Ratio".rjust(10) +  "   Prior"
    for k,v in sratios[0:top_n]:
        print k.ljust(80), str(round(v,4)).rjust(10) + " : " + str(prior_tally[k])

""" GET DATA """
def __get_data_docs__():
    import WordTokenizer
    import GwData as gd

    data = gd.GwData()
    docs = data.documents
    # Do NOT remove ANY stop words or LOW freq words, as we want tokens
    # to be sequential and patterns to include stop words if predictive
    tokenized_docs = \
        WordTokenizer.tokenize(docs, min_word_count=1, stem=True, remove_stop_words=False, spelling_correct=True)
    return data, tokenized_docs

""" HELPER """
def ratios_with_coverage(ratios, tally):
    computed = dict()
    for str_ngram, ratio in ratios:
        count = tally[str_ngram]
        computed[str_ngram] = ratio * 1.0 * count
    return sorted(computed.items(), key = lambda item: -item[1])

def merge(dict_a, dict_b):
    """
    Merge 2 dicts with same key, outputting the values as a tuple
    """
    result = dict()
    for akey in dict_a.keys():
        if akey in dict_b:
            result[akey] = (dict_a[akey], dict_b[akey])
    return result

def print_top_n_with_coverage(sorted_ratios, tally, n):
    merged = merge(dict(sorted_ratios), tally)
    # Sort by ratio desc
    s_merged = head(sorted(merged.items(), key=lambda item: -item[1][0]), n)

    for str_ngram, (ratio, cnt) in s_merged:
        print str_ngram.ljust(50), str(round(ratio,5)).ljust(10), cnt

""" GENERATE NGRAMS """
def predictive_ngrams(code, ngram_size_limit, min_doc_freq):
    data, tokenized_docs = __get_data_docs__()
    lbls = data.labels_for(code)
    return get_predictive_ngrams(tokenized_docs, lbls, compute_ngrams, ngram_size_limit, min_doc_freq)

def predictive_skipgrams(code, ngram_size_limit, min_doc_freq):
    data, tokenized_docs = __get_data_docs__()
    lbls = data.labels_for(code)
    return get_predictive_ngrams(tokenized_docs, lbls, compute_skip_grams, ngram_size_limit, min_doc_freq)

if __name__ == "__main__":

    sorted_positive_ratios, sorted_negative_ratios, positive_tally, negative_tally, total_freq, str2ngram = predictive_ngrams("50", 3, 5)
