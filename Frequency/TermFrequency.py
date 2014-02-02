from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
import logging, MatrixHelper, numpy as np

class TermFrequency(object):
    """ Computes a term frequency distance_matrix
    """
    def __init__(self, documents):
        logging.log(logging.INFO, "Creating Term Frequency")
        
        self.id2Word = Dictionary(documents)
        self.num_unique_words = len(self.id2Word)
        self.distance_matrix = self.to_term_frequency_matrix(documents)

    def to_term_frequency_vector(self, document):
        return self.id2Word.doc2bow(document)


    def to_binary_vector(self, document):
        tf = self.id2Word.doc2bow(document)
        vect = sparse2full(tf, len(self.id2Word.keys()))
        return np.array( vect > 0, dtype=int ) # concerts to binary

    def to_term_frequency_matrix(self, documents):
            return [self.to_term_frequency_vector(d) for d in documents]

    def binary_matrix(self):
        """ Turns a regular tf distance_matrix into a binary distance_matrix """
        def get_binary_data(val):
            if val <= 0:
                return 0
            return 1
       
        full_matrix = MatrixHelper.gensim_to_python_mdarray(self.distance_matrix, self.num_unique_words)
        return [[get_binary_data(cell)
                for cell in row]
                for row in full_matrix]
