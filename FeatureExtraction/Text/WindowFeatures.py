__author__ = 'simon.hughes'


def extract_positional_word_features(window, mid_ix, feature_val = 1):
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
    """

    feats = {}
    for i, (wd, tags) in enumerate(window):
        feature_name = "WD:" + str(-mid_ix + i) + " " + wd
        feats[feature_name] = feature_val

    return feats

def extract_word_features(window, feature_val = 1 ):
    """
    window      :   list of str
                        words in window
    feature_val :   Any
                        value for feature
    returns     :   dct
                        dct[str]:val

    Extracts positional word features
    """

    feats = {}
    for i, (wd, tags) in enumerate(window):
        feature_name = wd
        feats[feature_name] = feature_val
    return feats
