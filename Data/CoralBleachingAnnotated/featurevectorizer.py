__author__ = 'simon.hughes'

from sklearn.feature_extraction import DictVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from collections import defaultdict

class FeatureVectorizer(BaseEstimator, TransformerMixin):
    """ Class to filter features by frequency and vectorize
    """
    def __init__(self, min_feature_freq, sparse=False):
        """ Parameters
            ----------
            min_feature_freq : int
                Minimum feature frequency to retain to reduce dimensionality
        """
        self.min_feature_freq = min_feature_freq
        self.vectorizer = DictVectorizer(sparse=sparse)

    def fit(self, X, Y=None):
        """ Transform an array dictionaries into a numpy array

            Parameters
            ----------
            X : array-like - a list or tuple of dictionaries
                features to vectorize

            Returns
            -------
            X_new : numpy array of shape [n_samples, n_features_new]
                Transformed array
        """
        # Get items above the frequency threshold
        feature_freq = defaultdict(int)
        for dct in X:
            for k,v in dct.items():
                feature_freq[k] += 1
        self.frequent = set([k for k,v in feature_freq.items() if v >= self.min_feature_freq])

        # filter out low freq items
        filtered = []
        for dct in X:
            new_dct = dict()
            for k, v in dct.items():
                if k in self.frequent:
                    new_dct[k] = v
            #if len(new_dct) > 0: # need to keep to the same length as the ys array
            filtered.append(new_dct)
        # vectorize
        return self.vectorizer.fit(filtered)

    def transform(self, X):
        """
            Vectorizes an array of dictionary or map like objects
            Named features not encountered during fit or fit_transform will be
            silently ignored.

            Parameters
            ----------
            X : array-like list or tuple of dictionaries
        """
        return self.vectorizer.transform(X)