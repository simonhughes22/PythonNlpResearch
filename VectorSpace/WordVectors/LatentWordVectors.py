from collections import defaultdict

from gensim.models import LsiModel
from gensim.models import LdaModel
from gensim.models import RpModel
from MatrixHelper import unit_vector

import numpy as np
import TfIdf
import TermFrequency
from WindowSplitter import split_into_windows

class LatentWordVectors(object):

    def __init__(self, tokenized_docs, latentSpaceFactory, aggregation_method = "doc", normalize = False, unit_vectors = False, term_frequency_only = False):
        """
        Projects words to a vector space
        """
        tokenized_docs = [t for t in tokenized_docs if len(t) > 0]

        def pivot_by_words(dct, doc):
            for word1 in doc:
                for word2 in doc:
                    if word1 != word2:
                        dct[word1].append(word2)

        """ Pivot Docs Around Words """
        d = defaultdict(list)
        if aggregation_method == "doc":
            """ term - doc space """
            for i, doc in enumerate(tokenized_docs):
                for word in doc:
                    d[word].append(str(i))
        elif aggregation_method == "sentence":
            """ word space - words to words  """
            for i, doc in enumerate(tokenized_docs):
                self.pivot_by_words(d, doc)
        elif aggregation_method.startswith("window:"):
            _, str_size = aggregation_method.split(":")

            win_size = int(str_size)
            print "Window Size:", win_size

            win_id = 0
            for doc in tokenized_docs:
                windows = split_into_windows(doc, win_size)
                for win in windows:
                    for word in win:
                        d[word].append(str(win_id))
                        win_id += 1
                        #pivot_by_words(d, win)
            print "Size of windowed method:", len(d)
            pass
        else:
            raise Exception("Unexpected aggregation_method value: %s. Accepted Values are <'doc','sentence','window:n> " % aggregation_method)

        tokenized_docs = d.values()
        self.word_to_index = dict()
        for i, wd in enumerate(d.keys()):
            self.word_to_index[wd] = i

        if term_frequency_only:
            tf = TermFrequency.TermFrequency(tokenized_docs)
            latent_space = latentSpaceFactory(tf, tokenized_docs)
        else:
            tfidf = TfIdf.TfIdf(tokenized_docs)
            latent_space = latentSpaceFactory(tfidf, tokenized_docs)

        """ Construct Vector Space """
        self.latent_vector = []
        for i, v in enumerate(latent_space):
            vec = [val for idx, val in v]
            """ Example Normalization """
            if unit_vectors:
                vec = unit_vector(vec)
            self.latent_vector.append(vec)

        """ Normalize """
        if normalize:
            tmp_arr = np.array(self.latent_vector)
            means  = np.mean(tmp_arr, axis = 0)
            sds = np.std(tmp_arr, axis = 0)
            norm = (tmp_arr - means) / sds
            self.latent_vector = norm
        pass
        
    def project(self, item):
        if type(item) == type([]):
            return [self.project(t) for t in item]
        
        if item not in self.word_to_index:
            return None
        ix = self.word_to_index[item]
        return self.latent_vector[ix]

    @classmethod
    def TermFrequencySpace(cls, tokens, num_topics, aggregation_method = "doc", normalize = False, unit_vectors = False):
        def tf_fact(tfidf, tokenized_docs):
            return tfidf.distance_matrix
        return LatentWordVectors(tokens, tf_fact, aggregation_method = aggregation_method, normalize = normalize, unit_vectors = unit_vectors, term_frequency_only = True)

    """ Legacy for backwards compatibility """
    @classmethod
    def LsaSpace(cls, tokens, num_topics, aggregation_method="doc", normalize=False, unit_vectors=False):
        def lsa_fact(tfidf, tokenized_docs):
            corpus = LsiModel(corpus=tfidf.distance_matrix, id2word=tfidf.term_frequency.id2Word, num_topics=num_topics)
            return corpus[tfidf.distance_matrix]

        return LatentWordVectors(tokens, lsa_fact, aggregation_method=aggregation_method, normalize=normalize,
                                 unit_vectors=unit_vectors,term_frequency_only=False)
    """ END Legacy for backwards compatibility """

    @classmethod
    def LsaTfIdfSpace(cls, tokens, num_topics, aggregation_method = "doc", normalize = False, unit_vectors = False, chunk_size = 20000):
        def lsa_fact(tfidf, tokenized_docs):
            corpus = LsiModel(corpus = tfidf.distance_matrix, id2word = tfidf.term_frequency.id2Word, num_topics = num_topics, chunksize=chunk_size)
            return corpus[tfidf.distance_matrix]
        return LatentWordVectors(tokens, lsa_fact, aggregation_method = aggregation_method, normalize = normalize, unit_vectors = unit_vectors, term_frequency_only = False)

    @classmethod
    def LsaTermFrequencySpace(cls, tokens, num_topics, aggregation_method="doc", normalize=False, unit_vectors=False, chunk_size = 20000):
        def lsa_fact(tf, tokenized_docs):
            corpus = LsiModel(corpus=tf.distance_matrix, id2word=tf.id2Word, num_topics=num_topics, chunksize=chunk_size)
            return corpus[tf.distance_matrix]

        return LatentWordVectors(tokens, lsa_fact, aggregation_method=aggregation_method, normalize=normalize,
                                 unit_vectors=unit_vectors, term_frequency_only = True)

    @classmethod
    def RpTfIdfSpace(cls, tokens, num_topics, aggregation_method="doc", normalize=False, unit_vectors=False):
        def lsa_fact(tfidf, tokenized_docs):
            corpus = RpModel(corpus=tfidf.distance_matrix, id2word=tfidf.term_frequency.id2Word, num_topics=num_topics)
            return corpus[tfidf.distance_matrix]
        return LatentWordVectors(tokens, lsa_fact, aggregation_method=aggregation_method, normalize=normalize, unit_vectors=unit_vectors, term_frequency_only=False)

    @classmethod
    def RpTermFrequencySpace(cls, tokens, num_topics, aggregation_method="doc", normalize=False, unit_vectors=False):
        def lsa_fact(tf, tokenized_docs):
            corpus = RpModel(corpus=tf.distance_matrix, id2word=tf.id2Word, num_topics=num_topics)
            return corpus[tf.distance_matrix]

        return LatentWordVectors(tokens, lsa_fact, aggregation_method=aggregation_method, normalize=normalize,
                                 unit_vectors=unit_vectors, term_frequency_only=True)

    @classmethod
    def LdaSpace(cls, tokens, num_topics, aggregation_method = "doc", normalize = False, unit_vectors = False):
        def lda_fact(tfidf, tokenized_docs):
            model = LdaModel(corpus = tfidf.distance_matrix, id2word = tfidf.term_frequency.id2Word, num_topics = num_topics, update_every = 0, passes = 20)
            return model[tfidf.distance_matrix]
        return LatentWordVectors(tokens, lda_fact, aggregation_method = aggregation_method, normalize = normalize, unit_vectors = unit_vectors)
    
if __name__ == "__main__":
    
    import GwData
    import WordTokenizer
    
    data = GwData.GwData()
    tokens = WordTokenizer.tokenize(data.documents)
    
    lsa_v = LatentWordVectors.LsaTfIdfSpace(tokens, 100, True)
    
    pass