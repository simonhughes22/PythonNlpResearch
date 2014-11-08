__author__ = 'simon.hughes'

"""
    Feature extraction functions for working with word windws.
    All take a word window, the index of the middle word and a feature value.
    And return a feature dictionary (keys should be unique by feature)
"""

from NgramGenerator import compute_ngrams

def compute_middle_index(window):
    return int(round(len(window) / 2.0) - 1)

def extract_positional_word_features(window, mid_ix=None, feature_val = 1):
    """
    window      :   list of str
                        words in window
    mid_ix      :   int
                        position of word to predict
    feature_val :   Any
                        value for feature
    returns     :   dct
                        dct[str]:val

    Extracts positional word features
        (features are different for same word in different positions)
    """

    if mid_ix is None:
        mid_ix = compute_middle_index(window)
    feats = dict()
    for i, wd in enumerate(window):
        feature_name = "WD:" + str(-mid_ix + i) + " " + wd
        feats[feature_name] = feature_val

    return feats

def extract_word_features(window, mid_ix=None, feature_val = 1 ):
    """
    window      :   list of str
                        words in window
    mid_ix      :   int
                        position of word to predict (NOT USED - consistency)
    feature_val :   Any
                        value for feature
    returns     :   dct
                        dct[str]:val

    Extracts word features, IGNORING POSITION
    """

    feats = dict()
    for wd in window:
        feature_name = wd
        feats[feature_name] = feature_val
    return feats

def skip_b4_word_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts word features before the middle word, IGNORING POSITION
    """
    if mid_ix is None:
        mid_ix = compute_middle_index(window)
    feats = dict()
    target = window[mid_ix]
    for wd in window[:mid_ix]:
        feats["_BEFORE: " + wd + "|" + target] = feature_val
    return feats

def skip_after_word_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts word features after the middle word, IGNORING POSITION
    """
    if mid_ix is None:
        mid_ix = compute_middle_index(window)
    feats = dict()
    target = window[mid_ix]
    for wd in window[mid_ix+1:]:
        feats["_AFTER: " + target + "|" + wd] = feature_val
    return feats

def bigram_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts bi-gram word features, IGNORING POSITION
    """
    bi_grams = compute_ngrams(window, max_len = 2, min_len = 2)
    d = dict()
    for bi_gram in bi_grams:
        d["BI" + ":" + " " + bi_gram[0] + " | " + bi_gram[1]] = feature_val
    return d

def trigram_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts tri-gram word features, IGNORING POSITION
    """
    tri_grams = compute_ngrams(window, max_len = 3, min_len = 3)
    d = {}
    for tri_gram in tri_grams:
        tri_gram_key = tri_gram[0] + " | " + tri_gram[1] + "|" + tri_gram[2]
        d["TRI" + ":" + " " + tri_gram_key] = feature_val
    return d

def positional_bigram_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts bi-gram word features, INCLUDING POSITION
    """
    if mid_ix is None:
        mid_ix = compute_middle_index(window)
    bi_grams = compute_ngrams(window, max_len = 2, min_len = 2)
    d = {}
    for i, bi_gram in enumerate(bi_grams):
        d["P_BI" + ":" + str(-mid_ix + i) + " " + bi_gram[0] + " | " + bi_gram[1]] = feature_val
    return d

def positional_trigram_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts tri-gram word features, INCLUDING POSITION
    """
    if mid_ix is None:
        mid_ix = compute_middle_index(window)
    tri_grams = compute_ngrams(window, max_len = 3, min_len = 3)
    d = {}
    for i, tri_gram in enumerate(tri_grams):
        tri_gram_key = tri_gram[0] + " | " + tri_gram[1] + "|" + tri_gram[2]
        d["P_TRI" + ":" + str(-mid_ix + i) + " " + tri_gram_key] = feature_val
    return d

def positional_skip_word_features(window, mid_ix=None, feature_val = 1):
    """
        window      :   list of str
                            words in window
        mid_ix      :   int
                            position of word to predict
        feature_val :   Any
                            value for feature
        returns     :   dct
                            dct[str]:val

        Extracts skip grams including the target word, INCLUDING POSITION
    """
    if mid_ix is None:
        mid_ix = compute_middle_index(window)
    feats = {}
    target = window[mid_ix]
    for i, wd in enumerate(window):
        if i == mid_ix:
            continue
        a,b = wd,target
        if i > mid_ix:
            # swap
            a,b = b,a
        feats["P_SKIP:" + str(-mid_ix + i) + " " + a + " | " + b] = feature_val
    return feats
