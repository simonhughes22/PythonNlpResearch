__author__ = 'simon.hughes'

from sklearn.feature_extraction import DictVectorizer
from WindowFeatures import compute_middle_index
from collections import Counter

class WindowFeatureExtractor(object):
    """
    A simple wrapper class that takes a number of window based feature extractor
    functions and applies them to a dataset of windows, and then vectorizes with
    the sklearn DictVectorizer class
    """

    def __init__(self, feature_extractors, min_feat_frequency, sparse=True, feature_val=1):
        """
        feature_extractors  :   list of fns
                                    feature extraction fns
        min_feat_frequency  :   int
                                    minimum frequency of features to retain
        sparse              :   boolean
                                    return a sparse numpy matrix or not
        """
        self.feature_extractors = feature_extractors
        self.min_feat_frequency = min_feat_frequency
        self.vectorizer = DictVectorizer(sparse=sparse)
        self.feature_val = feature_val

    def fit(self, X, y=None):
        """
        X                   :   list of list of str
                                    list of word windows
        y                   :   ignored

        returns             :   numpy array (sparse is sparse = True)
        """
        feats = self.__extract_features_(X)
        return self.vectorizer.fit(feats)

    def transform(self, X, y=None):
        return self.vectorizer.transform(X, y)

    def fit_transform(self, X,y=None):
        feats = self.__extract_features_(X)
        return self.vectorizer.fit_transform(feats)

    def __extract_features_(self, X):
        if len(X) == 0:
            raise Exception("Empty list passed to WindowFeatureExtractor.fit")
        mid_ix = compute_middle_index(X[0])
        all_feats = []

        keys = []
        for window in X:
            d = {}
            for fn in self.feature_extractors:
                fts = fn(window, mid_ix, self.feature_val)
                d.update(fts)
            keys.extend(d.keys())
            all_feats.append(d)

        if self.min_feat_frequency <= 1:
            return all_feats

        """ Filter to at or above minimum feature frequency """
        keyCnt = Counter(keys)
        frequent = set([k for k,v in keyCnt.items() if v >= self.min_feat_frequency])

        freq_feats = []
        for d in all_feats:
            freq_d = dict([(k,v) for k,v in d.items() if k in frequent])
            freq_feats.append(freq_d)
        return freq_feats