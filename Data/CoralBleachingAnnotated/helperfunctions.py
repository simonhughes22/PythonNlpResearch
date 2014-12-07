__author__ = 'simon.hughes'
from collections import defaultdict
from IterableFP import flatten
import numpy as np

def get_word_feat_tags(essay_feats):
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

def get_ys_by_code(lst_tag_sets):

    unique_tags = set(flatten(lst_tag_sets))
    tmp_ys_bycode = defaultdict(list)
    for tag_set in lst_tag_sets:
        for y in unique_tags:
            tmp_ys_bycode[y].append(1 if y in tag_set else 0)

    ys_bycode = dict()
    for k, lst in tmp_ys_bycode.items():
        ys_bycode[y] = np.asarray(lst, dtype=np.int).reshape((1, len(lst)))
    return ys_bycode