from __future__ import absolute_import

import os
import random
import pickle
import logging

from collections import defaultdict
from perceptron import AveragedPerceptron
from results_procesor import compute_metrics
from Rpfa import weighted_mean_rpfa

PICKLE = "trontagger-0.1.0.pickle"

class PerceptronTaggerBinary(object):

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

    def __init__(self, classes, tag_history, load=False):
        self.tag_history = tag_history
        self.classes = set(classes)
        self.class2model = {}
        for cls in classes:
            self.class2model[cls] = AveragedPerceptron()
            self.class2model[cls].classes = set([self.NEGATIVE_CLASS, self.POSITIVE_CLASS])

    def _add_tag_features(self, feats, word, prev, prev2):
        sprev, sprev2 = str(prev), str(prev2)
        feats["bias"] = 1
        #feats["TAG -1 " + sprev]                  =      1     # included in other
        feats["TAG -1 wd " + sprev + "|" + word]  =      1
        #feats["TAG -2 " + sprev2]                 =      1     # included in other
        feats["TAG -2 wd " + sprev2 + "|" + word] =      1
        feats["TAG -1, -2 " + sprev + "|" + sprev2]=     1

    def _add_secondary_tag_features(self, feats, word, cls, prev_tags):
        for ix, prev in enumerate(prev_tags[-self.tag_history:]):
            offset = ix - self.tag_history
            if prev == self.POSITIVE_CLASS:
                feats["HIST_TAG " + str(offset) + " " + cls ]  = 1

    def predict(self, essay_feats):
        '''Tags a string `corpus`.'''

        # Assume untokenized corpus has \n between sentences and ' ' between words
        class2predictions = defaultdict(list)
        for essay_ix, essay in enumerate(essay_feats):
            for sent_ix, taggged_sentence in enumerate(essay.sentences):
                """ Start Sentence """
                class2prev = defaultdict(list)
                for cls in self.classes:
                    class2prev[cls] = list(self.START)

                for i, (wd) in enumerate(taggged_sentence):
                    # Don't mutate the feat dictionary
                    shared_features = dict(wd.features.items())
                    # get all tagger predictions for previous 2 tags
                    for cls in self.classes:
                        self._add_secondary_tag_features(shared_features, wd.word, cls, class2prev[cls])
                    # train each binary tagger
                    for cls in self.classes:
                        tagger_feats = dict(shared_features.items())
                        # add more in depth features for this tag
                        self._add_tag_features(tagger_feats, wd.word, class2prev[cls][-1], class2prev[cls][-2])
                        model = self.class2model[cls]
                        guess = model.predict(tagger_feats)
                        class2prev[cls].append(guess)
                        class2predictions[cls].append(guess)
        return class2predictions

    def __get_yal_(self, wd, tgt_tag):
        return self.POSITIVE_CLASS if tgt_tag in wd.tags else self.NEGATIVE_CLASS

    def train(self, essay_feats, save_loc=None, nr_iter=5):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''

        # Copy as we do an inplace shuffle below
        cp_essay_feats = list(essay_feats)

        for iter_ in range(nr_iter):
            class2predictions = defaultdict(list)
            class2tags = defaultdict(list)

            for essay_ix, essay in enumerate(cp_essay_feats):
                for sent_ix, taggged_sentence in enumerate(essay.sentences):
                    """ Start Sentence """
                    class2prev = defaultdict(list)
                    for cls in self.classes:
                        class2prev[cls] = list(self.START)

                    for i, (wd) in enumerate(taggged_sentence):
                        # Don't mutate the feat dictionary
                        shared_features = dict(wd.features.items())
                        # get all tagger predictions for previous 2 tags
                        for cls in self.classes:
                            self._add_secondary_tag_features(shared_features, wd.word, cls, class2prev[cls])
                        # train each binary tagger
                        for cls in self.classes:
                            tagger_feats = dict(shared_features.items())
                            # add more in depth features for this tag
                            self._add_tag_features(tagger_feats, wd.word, class2prev[cls][-1], class2prev[cls][-2])
                            actual = self.__get_yal_(wd, cls)
                            model = self.class2model[cls]
                            guess = model.predict(tagger_feats)
                            model.update(actual, guess, tagger_feats)

                            class2prev[cls].append(guess)

                            class2predictions[cls].append(guess)
                            class2tags[cls].append(actual)

            random.shuffle(cp_essay_feats)
            class2metrics = compute_metrics(class2tags, class2predictions)
            wtd_mean = weighted_mean_rpfa(class2metrics.values())
            logging.info("Iter {0}: Wtd Mean: {1}".format(iter_, str(wtd_mean)))

        for cls in self.classes:
            self.class2model[cls].average_weights()

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