from __future__ import absolute_import

import os
import random
import pickle
import logging

from collections import defaultdict
from perceptron import AveragedPerceptron
from results_procesor import ResultsProcessor
from Rpfa import weighted_mean_rpfa, micro_rpfa
import numpy as np

PICKLE = "trontagger-0.1.0.pickle"

class PerceptronTaggerMultiClassCombo(object):

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

    def __init__(self, individual_tags, tag_history, combo_freq_threshold, load=False, use_tag_features=True):
        self.use_tag_features = use_tag_features
        self.combo_freq_threshold = combo_freq_threshold
        self.tag_history = tag_history
        self.classes = set()
        self.individual_tags = set(individual_tags)

    def _add_tag_features(self, feats, word, prev, prev2):
        sprev, sprev2 = str(prev), str(prev2)
        feats["bias"] = 1
        # Commenting out the single previous tag features as included with the
        # tag history parameter
        #feats["TAG -1 " + sprev]                  =      1     # included in other
        feats["TAG -1 wd " + sprev + "|" + word]  =      1
        #feats["TAG -2 " + sprev2]                 =      1     # included in other
        feats["TAG -2 wd " + sprev2 + "|" + word] =      1
        feats["TAG -1, -2 " + sprev + "|" + sprev2]=     1

    def _add_secondary_tag_features(self, feats, prev_tags):
        for ix, prev in enumerate(prev_tags[-self.tag_history:]):
            offset = ix - self.tag_history
            feats["HIST_TAG " + str(offset) + " " + str(prev)] = 1

    def predict(self, essay_feats, output_scores = False):
        '''Tags a string `corpus`.
            Outputs a dictionary mapping to a list of binary predictions
        '''

        # Assume untokenized corpus has \n between sentences and ' ' between words
        class2predictions = defaultdict(list)
        for essay_ix, essay in enumerate(essay_feats):
            for sent_ix, taggged_sentence in enumerate(essay.sentences):
                """ Start Sentence """
                class2prev = defaultdict(list)
                for cls in self.classes:
                    class2prev[cls] = list(self.START)

                prev = list(self.START)
                for i, (wd) in enumerate(taggged_sentence):
                    # Don't mutate the feat dictionary
                    shared_features = dict(wd.features.items())
                    # get all tagger predictions for previous 2 tags

                    self._add_secondary_tag_features(shared_features, prev)

                    tagger_feats = dict(shared_features.items())
                    if self.use_tag_features:
                        self._add_tag_features(tagger_feats, wd.word, prev[-1], prev[-2])

                    scores_by_class = self.model.decision_function(tagger_feats)
                    guess = max(self.model.classes, key=lambda label: (scores_by_class[label], label))
                    prev.append(guess)

                    if output_scores:
                        max_score_per_class = defaultdict(float)
                        for fset_tags, score in scores_by_class.items():
                            for tag in fset_tags:
                                max_score_per_class[tag] = max(max_score_per_class[tag], score)

                        for cls in self.individual_tags:
                            class2predictions[cls].append(max_score_per_class[cls])
                    else:
                        for cls in self.individual_tags:
                            class2predictions[cls].append(1 if cls in guess else 0)

        np_class2predictions = dict()
        for key, lst in class2predictions.items():
            np_class2predictions[key] = np.asarray(lst)
        return np_class2predictions

    def decision_function(self, essay_feats):
        '''Tags a string `corpus`.
            Outputs a dictionary mapping to a list of scores for each class
        '''
        return self.predict(essay_feats, output_scores=True)

    def __get_tags_(self, tags):
        return frozenset((t for t in tags if t in self.individual_tags))

    def train(self, essay_feats, save_loc=None, nr_iter=5, verbose=True):
        '''Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        '''

        cp_essay_feats = list(essay_feats)

        # Copy as we do an inplace shuffle below
        tag_freq = defaultdict(int)
        for essay in cp_essay_feats:
            for taggged_sentence in essay.sentences:
                for wd in taggged_sentence:
                    fs_tags = self.__get_tags_(wd.tags)
                    tag_freq[fs_tags] +=1


        self.classes = set([ fs for fs, cnt in tag_freq.items() if cnt >= self.combo_freq_threshold])
        self.model = AveragedPerceptron(self.classes)

        for iter_ in range(nr_iter):
            class2predictions = defaultdict(list)
            class2tags = defaultdict(list)

            for essay_ix, essay in enumerate(cp_essay_feats):
                for sent_ix, taggged_sentence in enumerate(essay.sentences):
                    """ Start Sentence """
                    prev = list(self.START)

                    for i, (wd) in enumerate(taggged_sentence):
                        # Don't mutate the feat dictionary
                        shared_features = dict(wd.features.items())
                        # get all tagger predictions for previous 2 tags
                        self._add_secondary_tag_features(shared_features, prev)

                        tagger_feats = dict(shared_features.items())
                        # add more in depth features for this tag
                        actual = self.__get_tags_(wd.tags)

                        if self.use_tag_features:
                            self._add_tag_features(tagger_feats, wd.word, prev[-1], prev[-2])

                        guess = self.model.predict(tagger_feats)
                        self.model.update(actual, guess, tagger_feats)

                        prev.append(guess)
                        for cls in self.individual_tags:
                            class2predictions[cls].append(  1 if cls in guess  else 0 )
                            class2tags[cls].append(         1 if cls in actual else 0)

            random.shuffle(cp_essay_feats)
            class2metrics = ResultsProcessor.compute_metrics(class2tags, class2predictions)
            micro_metrics = micro_rpfa(class2metrics.values())
            if verbose:
                logging.info("Iter {0}: Micro Avg Metrics: {1}".format(iter_, str(micro_metrics)))

        self.model.average_weights()
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