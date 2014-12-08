__author__ = 'simon.hughes'
from collections import defaultdict
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
        for sent_ix, taggged_sentence in enumerate(essay):
            for word_ix, (wd) in enumerate(taggged_sentence):
                feats.append(wd.features)
                tags.append(wd.tags)
    return feats, tags

def get_wordlevel_ys_by_code(lst_tag_sets):
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
    for k, lst in tmp_ys_bycode.items():
        ys_bycode[k] = np.asarray(lst, dtype=np.int).reshape((len(lst), ))
    return ys_bycode

def train_wordlevel_classifier(xs, ysByCode, fn_create_cls, tags=None):
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
    tag2classifier = {}
    for code in sorted(tags):
        print "Training for :", code
        cls = fn_create_cls()
        tag2classifier[code] = cls
        ys = np.asarray(ysByCode[code])
        cls.fit(xs, ys)
    return tag2classifier

def __test_for_tag__(tag, xs, ysByCode, codeToClassifier):
    ys  = ysByCode[tag]
    pred_ys = codeToClassifier[tag].predict(xs)
    num_codes = len([y for y in ys if y == 1])
    r,p,f1,a = rpf1a(ys, pred_ys)
    return rpfa(r,p,f1,a,num_codes)

def test_word_level_classifiers(xs, ysByCode, tagToClassifier, tags=None):
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
    for tag in sorted(tags):
        cls = tagToClassifier[tag]
        metric = __test_for_tag__(tag, xs, ysByCode, tagToClassifier)
        metricsByTag[tag] = metric
        lst_metrics.append(metric)

    td_wt_mean_prfa   = weighted_mean_rpfa(lst_metrics)
    td_mean_prfa      = mean_rpfa(lst_metrics)
    return (metricsByTag, td_wt_mean_prfa, td_mean_prfa)

def print_metrics_for_codes(td_metricsByTag, vd_metricsByTag):

    def pad_str(val):
        return str(val).ljust(20)

    for tag in sorted(td_metricsByTag.keys()):
        td_rpfa = td_metricsByTag[tag]
        vd_rpfa = vd_metricsByTag[tag]
        print "TAG:       ", pad_str(tag)
        print "recall:    ", pad_str(td_rpfa.recall),       pad_str(vd_rpfa.recall)
        print "precision: ", pad_str(td_rpfa.precision),    pad_str(vd_rpfa.precision)
        print "f1:        ", pad_str(td_rpfa.f1_score),     pad_str(vd_rpfa.f1_score)
        print "accuracy:  ", pad_str(td_rpfa.accuracy),     pad_str(vd_rpfa.accuracy)
        print "sentences: ", pad_str(td_rpfa.num_codes),    pad_str(vd_rpfa.num_codes)
        print ""
