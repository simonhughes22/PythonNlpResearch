__author__ = 'simon.hughes'

from processessays import Essay

class Word(object):
    """ Holds a word for a sequence tagger approach.
    """
    def __init__(self, word, tags):
        """
        Holds a word and its associated tag(s)
        @param word: str
            word from a sentence
        @param tags: enumerable
            and enumerable of tags. Get's turned into a set of tags
        """
        self.word = word
        self.tags = set(tags)
        self.features = {}
        self.vector = None # for storing the final vector directly

    def __repr__(self):
        return self.word + "->" + str(self.tags)[3:] + " - %s feats" % str(len(self.features))

class FeatureExtractorInput(object):
    """ Holds all the input needed for a feature extractor
        wordix              :   int
                                    index of the current word in the sentence
        tagged_sentence     :   list of words
                                    sentence (words only)
        sentenceix          :   int
                                    index of the sentence in the essay
        essay               :   Essay class
                                    An essay name, and a list of tagged sentences,
                                    which are lists of tuples of words and a set of tags
    """
    def __init__(self, wordix, tagged_sentence, sentenceix, essay):
        self.wordix = wordix
        self.tagged_sentence = tagged_sentence
        self.sentenceix = sentenceix
        self.essay = essay
        # for convenience
        self.sentence, self.tags = zip(*tagged_sentence)
        # make hashable for memoization and immutable
        self.sentence = tuple(self.sentence)
        self.word = self.sentence[wordix]

class FeatureExtractorTransformer(object):
    def __init__(self, feature_extractor_fns):
        """ feature_extractor_fns   :   list of fns
                                            fn: FeatureExtractorInput -> dict
            tag_transformer         :   fn str -> str
                                            tranformation to apply to tags
            returns: a list of Essay objects
        """
        self.feature_extractor_fns = feature_extractor_fns

    def transform(self, essays):

        transformed = []
        for essay_ix, essay in enumerate(essays):
            t_essay = []
            transformed.append(Essay(essay.name, t_essay))
            for sent_ix, taggged_sentence in enumerate(essay.sentences):
                t_sentence = []
                t_essay.append(t_sentence)

                for word_ix, (wd, tags) in enumerate(taggged_sentence):
                    word = Word(wd, tags)
                    input = FeatureExtractorInput(word_ix, taggged_sentence, sent_ix, essay)
                    for fn in self.feature_extractor_fns:
                        d = fn(input)
                        word.features.update(d)
                    t_sentence.append(word)
        return transformed

