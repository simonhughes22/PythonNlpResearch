__author__ = 'simon.hughes'

import numpy as np

class Word(object):
    """ Holds a word for a sequence tagger approach.
    """
    def __int__(self, word, tags):
        self.word = word
        self.tags = set(tags)
        self.features = {}

class FeatureExtractorInput(object):
    """ Holds all the input needed for a feature extractor

        word        :   Word class
                            current word + tags + features
        wordix      :   int
                            index of word in the sentence
        sentence    :   list of words
                            sentence (words only)
        sentenceix  :   int
                            index of the sentence in the essay
        essay       :   list of list of tuples
                            list of sentences, which are lists of
                            tuples of words and a set of tags
    """
    def __int__(self, word, wordix, sentence, sentenceix, essay):
        self.word = word
        self.wordix = wordix
        self.sentence = sentence
        self.sentenceix = sentenceix
        self.essay = essay

class FeatureExtractor(object):
    def __int__(self, feature_extractor_fns):
        """ feature_extractor_fns   :   list of fns
                                            fn: FeatureExtractorInput -> dict
        """
        self.feature_extractor_fns = feature_extractor_fns

    def transform(self, essays):

        transformed = []
        for essay_ix, essay in enumerate(essays):
            t_essay = []
            transformed.append(t_essay)
            for sent_ix, taggged_sentence in enumerate(essay):
                t_sentence = []
                sentence = zip(*taggged_sentence)
                for word_ix, (wd, tags) in taggged_sentence:
                    word = Word(wd, tags)
                    input = FeatureExtractorInput(word, word_ix, sentence, sent_ix, essay)
                    for fn in self.feature_extractor_fns:
                        d = fn(input)
                        word.features.update(d)
                    t_sentence.append(word)
        return transformed

