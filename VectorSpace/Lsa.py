from gensim.models import LsiModel
import logging

class Lsa(object):
    """ Takes a tf idf object
        and projects into LSA space
    """

    def __init__(self, tfidf, num_topics = 100):
        logging.log(logging.INFO, "Creating LSA Matrix with {0} dimensions".format(num_topics))
        
        self.__lsa__ = LsiModel(corpus = tfidf.distance_matrix, id2word = tfidf.term_frequency.id2Word, num_topics = num_topics)
        
        self.tfidf = tfidf
        self.num_topics = num_topics
        self.distance_matrix = self.__lsa__[tfidf.distance_matrix]
        self.id2Word = tfidf.id2Word

    def to_lsa_vector(self, tokenized_doc):
        tfidf_vector = self.tfidf.to_tfidf_vector(tokenized_doc)
        return self.__lsa__[tfidf_vector]

    def to_lsa_matrix(self, tokenized_docs):
        tfidf_matrix = self.tfidf.to_tfidf_matrix(tokenized_docs)
        return self.__lsa__[tfidf_matrix]
