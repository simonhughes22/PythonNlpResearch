import IdGenerator
import numpy as np
from featureextractortransformer import FeatureExtractorInput, Word

class WordProjectorTransformer(object):


    def __init__(self, offset):
        self.offset = offset
        self.idgen = IdGenerator.IdGenerator()

    def transform(self, essays):
        for essay_ix, essay in enumerate(essays):
            for sent_ix, taggged_sentence in enumerate(essay):
                for word_ix, (wd, tags) in enumerate(taggged_sentence):
                    _ = self.idgen.get_id(wd)

        self.total_num_features = self.idgen.max_id() + 1

        self.START = np.zeros(self.total_num_features)
        self.END   = np.zeros(self.total_num_features)

        transformed = []
        for essay_ix, essay in enumerate(essays):
            t_essay = []
            transformed.append(t_essay)
            for sent_ix, taggged_sentence in enumerate(essay):
                t_sentence = []
                t_essay.append(t_sentence)

                for word_ix, (wd, tags) in enumerate(taggged_sentence):
                    feat_input = FeatureExtractorInput(word_ix, taggged_sentence, sent_ix, essay)
                    word = Word(wd, tags)
                    word.vector = self.__get_window_(input=feat_input)
                    t_sentence.append(word)
        return transformed

    def __word2vector_(self, wd):
        id = self.idgen.get_id(wd)
        vect = np.zeros(self.total_num_features)
        vect[id] = float(1.0)
        return vect

    def __get_window_(self, input):

        start = input.wordix - self.offset
        stop = input.wordix + self.offset

        end = len(input.sentence) - 1
        vecs = []
        for i in range(start, stop + 1):
            relative_offset = str(i - input.wordix)
            if i < 0:
                vecs.append(self.START.copy())
            elif i > end:
                vecs.append(self.END.copy())
            else:
                offset_word = input.sentence[i]
                vecs.append(self.__word2vector_(offset_word))

        # stack vectors column wise to form a long column vector
        return np.hstack(vecs)