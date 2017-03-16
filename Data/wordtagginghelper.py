__author__ = 'simon.hughes'
from collections import defaultdict, OrderedDict

import numpy as np
import scipy

from IterableFP import flatten
from processessays import Essay
from featureextractortransformer import Word


def flatten_to_wordlevel_feat_tags(essay_feats):
    """
    Splits the essay-level features into a list of word level features for tagging

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

def flatten_to_wordlevel_vectors_tags(essay_feats, sparse=True):
    """
    Splits the essay-level features into

    Parameters
    ----------
    essay_feats : a list of lists of lists of Word objects
        Tag level features for the essays
    sparse      : boolean - return sparse vectors or not?

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

    if sparse:
        return scipy.sparse.csr_matrix(temp_features), tags
    else:
        return np.asarray(temp_features), tags

def get_wordlevel_ys_by_code(lst_tag_sets, expected_tags):
    """
    Convert a list of tagsets to a dictionary of ys values per tag label

    Parameters
    ----------
    lst_tag_sets : a list of sets of tags
        List of labels for each word

    Returns
    ----------
    A dictionary of codes mapping to binary labels for that code

    """
    unique_tags = set(flatten(lst_tag_sets))
    tmp_ys_bycode = defaultdict(list)
    for tag_set in lst_tag_sets:
        for y in unique_tags:
            tmp_ys_bycode[y].append(1 if y in tag_set else 0)

    num_rows = len(tmp_ys_bycode.values()[0])

    # Convert to a dict of numpy arrays
    ys_bycode = dict()
    for tag in expected_tags:
        if tag in tmp_ys_bycode and len(tmp_ys_bycode[tag]) > 0:
            lst = tmp_ys_bycode[tag]
            ys_bycode[tag] = np.asarray(lst, dtype=np.int).reshape((len(lst), ))
        else:
            ys_bycode[tag] = np.zeros(shape=(num_rows,), dtype=np.int)
    return ys_bycode

def get_wordlevel_powerset_ys(lst_tag_sets, expected_tags):
    """
    Convert a list of tagsets to a dictionary of ys values per tag label

    Parameters
    ----------
    lst_tag_sets : a list of sets of tags
        List of labels for each word

    expected_tags : array like
        tags we care about

    Returns
    -----------
    list of labels
    """

    expected_tags = set(expected_tags)

    ys = []
    for tag_set in lst_tag_sets:
        isect = expected_tags.intersection(tag_set)
        if isect:
            lbl = ",".join(sorted(isect))
        else:
            lbl = "O"
        ys.append(lbl)
    return ys

def get_wordlevel_mostfrequent_ys(lst_tag_sets, expected_tags, tag_freq):
    """
    Convert a list of tagsets to a dictionary of ys values per tag label

    Parameters
    ----------
    lst_tag_sets : a list of sets of tags
        List of labels for each word

    expected_tags : array like
        tags we care about

    tag_freq : dict[str] -> int
        frequency of tags

    Returns
    -----------
    list of labels
    """

    expected_tags = set(expected_tags)

    ys = []
    for tag_set in lst_tag_sets:
        isect = expected_tags.intersection(tag_set)
        if len(isect) > 0:
            most_common = max(isect, key=lambda tag: tag_freq[tag])
            lbl = most_common
        else:
            lbl = "O"
        ys.append(lbl)
    return ys

def get_wordlevel_ys_by_labelpowerset(lst_tag_sets, expected_tags, min_powerset_freq):
    """
    Convert a list of tagsets to a dictionary of ys values per tag label

    Parameters
    ----------
    @param lst_tag_sets : a list of sets of tags
        List of labels for each word

    @param min_powerset_freq: int
        Minimum (word) frequency required for each label powerset to be considered

    Returns
    ----------
    @rtype: defaultdict(np.array)

    """
    sexpected_tags = set(expected_tags)
    powerset_freq = defaultdict(int)
    lst_powersets = []
    for tag_set in lst_tag_sets:
        tags = frozenset((t for t in tag_set if t in sexpected_tags))
        powerset_freq[tags] += 1
        lst_powersets.append(tags)

    freq_powersets = set((t for t, freq in powerset_freq.items()
                          # only psets above threshold, and non empty ones
                          if freq >= min_powerset_freq and len(t) > 0))

    tmp_ys_bypowerset = defaultdict(list)
    for tag_set in lst_powersets:
        for pset in freq_powersets:
            tmp_ys_bypowerset[pset].append(1 if tag_set == pset else 0)

    ys_bypowerset = dict()
    for pset in freq_powersets:
        lst = tmp_ys_bypowerset[pset]
        ys_bypowerset[pset] = np.asarray(lst, dtype=np.int).reshape((len(lst), ))
    return ys_bypowerset

def get_wordlevel_predictions_by_code_from_powerset_predictions(ys_bypowerset, expected_tags):

    """

    Parameters
    -----------
    @param ys_bypowerset: defaultdict[frozenset[str], np.array]
        An array of predictions per label powerset
    @param expected_tags: list[str]
        Expected individual labels (powerset are powersets of this set)

    Return
    ----------
    @return: defaultdict[str, np.array]
    """

    rows = len(ys_bypowerset.values()[0])
    ys_by_code = defaultdict(lambda : np.zeros(shape=(rows,), dtype=np.int))

    for pset, pred in ys_bypowerset.items():
        if type(pred) != np.ndarray:
            pred = np.asarray(pred).reshape((rows, ))
        # Add all predictions for a single code together
        for y in pset:
            ys_by_code[y] += pred

    for tag in expected_tags:
        # force creation of np.zeros if tag is not present
        predictions = ys_by_code[tag]
        # where the same code is predicted by more than one classifier, set to 1
        predictions[predictions > 1] = 1
    return ys_by_code

def get_by_code_from_powerset_predictions(predictions, expected_tags):
    pred_by_code = defaultdict(list)
    for pred in predictions:
        pred_labels = set(pred.split(","))
        for lbl in expected_tags:
            if lbl in pred_labels:
                pred_by_code[lbl].append(1)
            else:
                pred_by_code[lbl].append(0)
    return pred_by_code

class always_false(object):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.zeros((x.shape[0],), dtype=np.int)

    def predict_proba(self, x):
        return np.zeros((x.shape[0],), dtype=np.float64)

    def decision_function(self, x):
        return -1.0 * np.ones((x.shape[0],), dtype=np.float64)

#TODO Parallelize
def train_classifier_per_code(xs, ysByCode, fn_create_cls, tags=None, verbose=True):
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
        if verbose:
            print("Training for :" + code)
        ys = np.asarray(ysByCode[code])
        if len(ys) == 0 or max(ys) == 0:
            cls = always_false()
        else:
            cls = fn_create_cls()
            cls.fit(xs, ys)
        tag2classifier[code] = cls
    return tag2classifier

def merge_dictionaries(dictFrom, dictTo):
    """
    Appends a dict[str]: list to an existing defaultdict[str]: list

    @param dictFrom: dict[str] : numpyarray
    @param dictTo:   defaultdict[str]: list

    @return: dictTo
    """

    for k, lst in dictFrom.items():
        dictTo[k].extend(lst)
    return dictTo

def predict_for_tag(tag, xs, codeToClassifier):
    return codeToClassifier[tag].predict(xs)

def probability_for_tag(tag, xs, codeToClassifier):
    # This causes constant issues. Sometimes the output is of one shape, sometimes another
    # The main issue is that it outputs 2 probabilities, one per class. We just want to take one of those
    #return codeToClassifier[tag].predict_proba(xs)[0]

    pred = codeToClassifier[tag].predict_proba(xs)
    # 2D output
    if len(pred.shape) == 2:
        # Works for /Users/simon.hughes/GitHub/NlpResearch/PythonNlpResearch/Experiments/GlobalWarmingAnnotated/WindowBasedClassifier/WindowBased_Classifier.py
        # WHEN USE_SVM = False (i.e. use Log Regression)
        return pred[:,0]
    return pred

def decision_function_for_tag(tag, xs, codeToClassifier):
    return codeToClassifier[tag].decision_function(xs)

def test_classifier_per_code(xs, tagToClassifier, tags=None, predict_fn=predict_for_tag):
    """
    Compute metrics over tagging data

    Parameters
    ----------
    xs : numpy array
        Features over the words
    tag2classifier : dict[str]:BaseEstimator
        A dictionary mapping tags to classifiers
    tags : list(str)
        List of tags to test over. Use ysByCode.keys() if none
    predict_fn : (tag,xs,codeToClassifier) => np.array
        A function to predict the labels for a given tag
    Returns
    -------
    predictions_by_code : a pair of dict's mapping tags to their actual labels \ predictions
    """
    if tags == None:
        tags = tagToClassifier.keys()

    predictions_by_code = dict()
    for tag in sorted(tags):
        pred_ys = predict_fn(tag, xs, tagToClassifier)
        predictions_by_code[tag] = pred_ys

    return predictions_by_code

def essaysfeats_to_most_common_tags(essay_feats, tag_freq):
    """
    Returns a set of essay features with the labels filtered to the most frequent labels

    Parameters
    ----------
    essay_feats : list[Essay]
        a list of Essay objects objects/
    tag_freq : dict[str, int]
        a dictionary mapping each tag to its frequency in the dataset

    Returns
    -------
    essay_feats : list[Essay]
    """

    valid_tags = set(tag_freq.keys())

    new_essay_feats = []
    for essay_ix, essay in enumerate(essay_feats):

        new_sentences = []
        new_essay = Essay(essay.name, sentences=new_sentences)
        new_essay_feats.append(new_essay)

        # The essay class is used in 2 different variants. Here, the
        # sentences are lists of Word objects
        for sent_ix, taggged_sentence in enumerate(essay.sentences):

            new_sentence = []
            new_sentences.append(new_sentence)
            for word_ix, (wd) in enumerate(taggged_sentence):

                fltrd_tags = set()
                # if we have tags, filter to most frequent recognized tag
                if len(wd.tags) > 0:
                    # need to filter to just those that are in tag_freq
                    recognized_tags = valid_tags.intersection(wd.tags)
                    if len(recognized_tags) > 0:
                        most_frequent = max(recognized_tags, key=lambda tag: tag_freq[tag])
                        fltrd_tags.add(most_frequent)

                new_word = Word(wd.word, fltrd_tags)
                # make sure to copy over the features
                new_word.features = dict(wd.features)
                # leave the vector as None (this isn't used I don't think)
                new_sentence.append(new_word)

    return new_essay_feats