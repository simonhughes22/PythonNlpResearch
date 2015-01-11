__author__ = 'simon.hughes'
from collections import defaultdict, OrderedDict
from IterableFP import flatten
from Metrics import rpf1a
from Rpfa import rpfa, weighted_mean_rpfa, mean_rpfa
import numpy as np

def flatten_to_wordlevel_feat_tags(essay_feats):
    """
    Splits the essay-level features into

    Parameters
    ----------
    essay_feats : a list of lists of lists of Word objects
        Tag level features for the essays

    Returns
    -------
    feats, tags : a 2 tuple of a list of feature dictionaries and a list of sets of tags
        The flattened features and tags from the essay words
    """
    feats = []
    tags = []
    for essay_ix, essay in enumerate(essay_feats):
        for sent_ix, taggged_sentence in enumerate(essay.sentences):
            for word_ix, (wd) in enumerate(taggged_sentence):
                feats.append(wd.features)
                tags.append(wd.tags)
    return feats, tags

def flatten_to_wordlevel_vectors_tags(essay_feats):
    """
    Splits the essay-level features into

    Parameters
    ----------
    essay_feats : a list of lists of lists of Word objects
        Tag level features for the essays

    Returns
    -------
    feats, tags : a 2 tuple of a list of feature dictionaries and a list of sets of tags
        The flattened features and tags from the essay words
    """
    tags = []
    temp_features = []
    for essay_ix, essay in enumerate(essay_feats):
        for sent_ix, taggged_sentence in enumerate(essay):
            for word_ix, (wd) in enumerate(taggged_sentence):
                temp_features.append(wd.vector)
                tags.append(wd.tags)
    return np.asarray(temp_features), tags

def get_wordlevel_ys_by_code(lst_tag_sets, expected_tags):
    """
    Convert a list of tagsets to a dictionary of ys values per tag label

    Parameters
    ----------
    lst_tag_sets : a list of sets of tags
        List of labels for each word

    """
    unique_tags = set(flatten(lst_tag_sets))
    tmp_ys_bycode = defaultdict(list)
    for tag_set in lst_tag_sets:
        for y in unique_tags:
            tmp_ys_bycode[y].append(1 if y in tag_set else 0)

    ys_bycode = dict()
    for tag in expected_tags:
        if tag in tmp_ys_bycode:
            lst = tmp_ys_bycode[tag]
            ys_bycode[tag] = np.asarray(lst, dtype=np.int).reshape((len(lst), ))
        else:
            ys_bycode[tag] = np.zeros(shape=(len(tmp_ys_bycode.values()[0]), ), dtype=np.int)
    return ys_bycode


class always_false(object):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.zeros((x.shape[0],), dtype=np.int)

    def predict_proba(self, x):
        return np.zeros((x.shape[0],), dtype=np.float64)

#TODO Parallelize
def train_classifier_per_code(xs, ysByCode, fn_create_cls, tags=None):
    """
    Trains an instance of the classifier per code in codes

    Parameters
    ----------
    xs : numpy array
        Features for each word to be tagged
    ysByCode : A dict of labels to np arrays
        Ys values partitioned by label
    fn_create_cls : function ()-> BaseEstimator
        factory function to create the classifier
    tags : (optional) a collection of str
        Tags to classify - can be used to filter ysByCode

    Returns
    -------
    tag2classifier: dict[str]:BaseEstimator
        A dictionary mapping each tag to a classifier trained on that tag

    """

    if tags == None:
        tags = ysByCode.keys()
    tag2classifier = OrderedDict()
    for code in sorted(tags):
        print "Training for :", code
        ys = np.asarray(ysByCode[code])
        if max(ys) == 0:
            cls = always_false()
        else:
            cls = fn_create_cls()
            cls.fit(xs, ys)
        tag2classifier[code] = cls
    return tag2classifier

def __test_for_tag__(tag, xs, ysByCode, codeToClassifier):
    ys  = ysByCode[tag]
    pred_ys = codeToClassifier[tag].predict(xs)
    num_codes = len([y for y in ys if y == 1])
    r,p,f1,a = rpf1a(ys, pred_ys)
    return rpfa(r,p,f1,a,num_codes), pred_ys

def test_classifier_per_code(xs, ysByCode, tagToClassifier, tags=None):
    """
    Compute metrics over tagging data

    Parameters
    ----------
    xs : numpy array
        Features over the words
    ysByCode : dict[str]:(numpy array)
        Dictionary mapping tags to binary labels
    tag2classifier : dict[str]:BaseEstimator
        A dictionary mapping tags to classifiers

    Returns
    -------
    metricsByTag, td_wt_mean_prfa, td_mean_prfa : a 3-tuple of

        a dictionary storing rfpa values for each tag
        the mean weighted recall precision and accuracy metric
        the mean recall precision and accuracy metric

    """
    if tags == None:
        tags = ysByCode.keys()
    lst_metrics = []
    metricsByTag = dict()
    predictions_by_code = dict()
    for tag in sorted(tags):
        metric, pred_ys = __test_for_tag__(tag, xs, ysByCode, tagToClassifier)
        metricsByTag[tag] = metric
        lst_metrics.append(metric)
        predictions_by_code[tag] = pred_ys

    td_wt_mean_prfa   = weighted_mean_rpfa(lst_metrics)
    td_mean_prfa      = mean_rpfa(lst_metrics)
    return (metricsByTag, td_wt_mean_prfa, td_mean_prfa, predictions_by_code)

