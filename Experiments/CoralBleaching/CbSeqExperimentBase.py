__author__ = 'simon.hughes'

from collections import defaultdict
from XmlEssay import essay_xml_loader, extract_tagged_sentences
from nltk import PorterStemmer, tokenize as tkn, flatten
from SpellingCorrector import SpellingCorrector
from abc import ABCMeta, abstractmethod
from CrossValidation import cross_validation
import numpy as np

class CbSeqExperimentBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.positive_label =  1
        self.negative_label = -1

    def load_tagged_sentences(self):
        essays = essay_xml_loader()
        return extract_tagged_sentences(essays)

    def extract_tag(self, tags, tag):
        return [self.positive_label if t == tag else self.negative_label
                for t in tags]

    def process_sentences(self, sentences, spelling_correct, stem, min_wd_cnt):
        def compose(fna, fnb):
            def comp(wd):
                a_result = fna(wd)
                return fnb(a_result)

            return comp

        map_fn = lambda i: i
        LOW_FREQ_WD = "LOW_FREQ"
        if stem:
            stemmer = PorterStemmer()
            map_fn = compose(map_fn, stemmer.stem)
        if spelling_correct:
            corrector = SpellingCorrector()
            map_fn = compose(map_fn, corrector.correct)
        word_freq = defaultdict(int)
        processed_sentences = []
        for sent in sentences:
            processed_sent = []
            for wd in sent:
                mapped = map_fn(wd)
                word_freq[mapped] += 1
                processed_sent.append(mapped)
            processed_sentences.append(processed_sent)

        def replace_low_freq_words(wd):
            if word_freq[wd] < min_wd_cnt:
                return LOW_FREQ_WD
            return wd

        return map(lambda sent: map(replace_low_freq_words, sent), processed_sentences)

    def tags_for_code(self, code, tagged_sentences):
        code_tags = []
        for tagged_sentence in tagged_sentences:
            tags = self.extract_tag(tagged_sentence, code)
            code_tags.append(tags)
        return code_tags

    def run(self, min_wd_cnt = 5, stem = True, spelling_correct = True, folds = 10):

        """ We don't want to remove stop words
        """
        sentences, tagged_sentences = self.load_tagged_sentences()
        processed_sentences = self.process_sentences(sentences, spelling_correct, stem, min_wd_cnt)
        sentence_features = np.asarray( map(self.features_for_sentence, processed_sentences))

        cross_validation_ixs = cross_validation(range(len(sentences)), folds)
        codes = sorted(set(flatten(tagged_sentences)))

        for code in codes:

            code_tags = self.tags_for_code(code, tagged_sentences)

            pass

        pass

    @abstractmethod
    def features_for_sentence(self, sentence):
        return NotImplemented
