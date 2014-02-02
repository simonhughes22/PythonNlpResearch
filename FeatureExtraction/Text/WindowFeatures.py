__author__ = 'simon.hughes'

SENTENCE_START  = "SENTENCE_START"
SENTENCE_END    = "SENTENCE_END"

def add_bookends(sentence, tags):
    """
    sentence    :   list of str
                        sentence
    tags        :   list of str
                        tags for sentence
    returns (padded sentence, padded tags)

    Adds special start and end tags to a sentence
    """
    return (
        [SENTENCE_START]  + sentence + [SENTENCE_END],
        [None] + tags + [None]
    )

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
