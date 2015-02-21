from __future__ import absolute_import
import os
import random
from collections import defaultdict
import pickle
import logging

from textblob.base import BaseTagger
from textblob.tokenizers import WordTokenizer, SentenceTokenizer
from perceptron import AveragedPerceptron

PICKLE = "trontagger-0.1.0.pickle"

class PerceptronTagger(object):

    '''Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
        http://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
    :param load: Load the pickled model upon instantiation.
    '''

    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), PICKLE)

    POSITIVE_CLASS = 1.0
    NEGATIVE_CLASS = 0.0

    def __init__(self, target_tag, load=False):
        self.target_tag = target_tag
        self.model = AveragedPerceptron()

        self.classes = set([self.POSITIVE_CLASS, self.NEGATIVE_CLASS])
        if load:
            self.load(self.AP_MODEL_LOC)

    def _add_tag_features(self, feats, word, prev, prev2):
        sprev, sprev2 = str(prev), str(prev2)
        feats["TAG -1" + sprev]                  =      1
        feats["TAG -1 wd" + sprev + "|" + word]  =      1
        feats["TAG -2 " + sprev2]                =      1
        feats["TAG -2 wd" + sprev2 + "|" + word] =      1
        feats["TAG -1, -2" + sprev + "|" + sprev2]=     1

    def predict(self, essay_feats):
        '''Tags a string `corpus`.'''
        # Assume untokenized corpus has \n between sentences and ' ' between words

        predictions = []

        for essay_ix, essay in enumerate(essay_feats):
            for sent_ix, taggged_sentence in enumerate(essay.sentences):
                prev, prev2 = self.START
                for i, (wd) in enumerate(taggged_sentence):

                    features = wd.features
                    self._add_tag_features(features, wd.word, prev, prev2)
                    tag = self.model.predict(features)
                    predictions.append(tag)
                    prev2 = prev
                    prev = tag
        return predictions

    def __get_yal_(self, wd):
        return self.POSITIVE_CLASS if self.target_tag in wd.tags else self.NEGATIVE_CLASS

    def train(self, essay_feats, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''

        # Copy as we do an inplace shuffle below
        cp_essay_feats = list(essay_feats)
        self.model.classes = self.classes
        prev, prev2 = self.START
        for iter_ in range(nr_iter):
            c = 0
            n = 0

            for essay_ix, essay in enumerate(cp_essay_feats):
                for sent_ix, taggged_sentence in enumerate(essay.sentences):
                    prev, prev2 = self.START
                    for i, (wd) in enumerate(taggged_sentence):
                        # Don't mutate the feat dictionary
                        features = dict(wd.features.items())
                        self._add_tag_features(features, wd.word, prev, prev2)
                        actual = self.__get_yal_(wd)
                        guess = self.model.predict(features)
                        self.model.update(actual, guess, features)
                        prev2 = prev
                        prev = guess
                        c += guess == actual
                        n += 1
            random.shuffle(cp_essay_feats)
            logging.info("Iter {0}: {1}/{2}={3}% correct".format(iter_, c, n, _pc(c, n)))
        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                         open(save_loc, 'wb'), -1)
        return None

    def load(self, loc):
        '''Load a pickled model.'''
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            msg = ("Missing trontagger.pickle file.")
            raise Exception(msg)
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None

    def _normalize(self, word):
        '''Normalization used in pre-processing.
        - All words are lower cased
        - Digits in the range 1800-2100 are represented as !YEAR;
        - Other digits are represented as !DIGITS
        :rtype: str
        '''
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        '''Map tokens into a feature representation, implemented as a
        {hashable: float} dict. If the features change, a new model must be
        trained.
        '''
        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)
        # It's useful to have a constant feature, which acts sort of like a prior
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0])
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i-1])
        add('i-1 suffix', context[i-1][-3:])
        add('i-2 word', context[i-2])
        add('i+1 word', context[i+1])
        add('i+1 suffix', context[i+1][-3:])
        add('i+2 word', context[i+2])
        return features

def _pc(n, d):
    return (float(n) / d) * 100