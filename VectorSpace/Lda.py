from gensim.models import LdaModel
import logging

class Lda(object):
    """ Takes a tf idf object
        and projects into LSA space
    """

    def __init__(self, tfidf, num_topics = 100):
        logging.log(logging.INFO, "Creating LDA Topic Model with {0} topics".format(num_topics))
        
        self.__lsa__ = LdaModel(
                                corpus = tfidf.distance_matrix, 
                                id2word = tfidf.term_frequency.id2Word, 
                                num_topics = num_topics,
                                update_every = 0,
                                passes = 20)
        
        self.tfidf = tfidf
        self.num_topics = num_topics
        self.distance_matrix = self.__lsa__[tfidf.distance_matrix]

    def to_lda_vector(self, sentence):
        tfidf_vector = self.tfidf.to_tfidf_vector(sentence)
        return self.__lsa__[tfidf_vector]
