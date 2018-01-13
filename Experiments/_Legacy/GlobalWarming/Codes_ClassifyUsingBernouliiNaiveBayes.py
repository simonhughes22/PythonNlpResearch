
from sklearn.naive_bayes import BernoulliNB
from GwExperimentBase import *


class Codes_ClassifyUsingBernouliiNaiveBayes(GwExperimentBase):

    def __init__(self):
        pass

    def get_vector_space(self, tokenized_docs):
        return self.term_freq_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            cls = BernoulliNB()
            cls.fit(xs, ys.tolist())
            return cls
        return cls_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        #return None
        return Converter.to_binary

if __name__ == "__main__":

    #C = 1.0

    cl = Codes_ClassifyUsingBernouliiNaiveBayes()
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingBernouliiNaiveBayes.txt", min_word_count = 5)
            
    