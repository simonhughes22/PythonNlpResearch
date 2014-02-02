from gensim.models import TfidfModel

import logging
import TermFrequency as tf

class TfIdf(object):
    """ Takes a list of documents
        and computes a tf idf distance_matrix 
    """
    def __init__(self, documents):
        
        logging.log(logging.INFO, "Creating TfIdf")

        self.term_frequency = tf.TermFrequency(documents)
        self.id2Word = self.term_frequency.id2Word
      
        self.__tf_Matrix__ = self.term_frequency.distance_matrix
        self.__tfidf_model__ = TfidfModel(self.__tf_Matrix__)

        # Load into RAM - Note for large corpora, you won't want to do this
        # Instead, compute once and dump to disk
        #   then load into your own corpus object and iterate through it
        self.distance_matrix = self.__tfidf_model__[self.__tf_Matrix__]

    def to_tfidf_vector(self, tokenized_doc):
        tf_vector = self.id2Word.doc2bow(tokenized_doc) 
        return self.__tfidf_model__[tf_vector]
    
    def to_tfidf_matrix(self, tokenized_docs):
        tf_matrix = [self.id2Word.doc2bow(r) 
                     for r in tokenized_docs] 
        return self.__tfidf_model__[tf_matrix]


