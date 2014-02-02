from sklearn.lda import LDA
from GwExperimentBase import *
import numpy as np

class Codes_ClassifyUsingLDA_WithLSAValues(GwExperimentBase):

    def __init__(self):
        self.num_topics = 500
        pass #no params     

    def get_vector_space(self, tokenized_docs):
        return self.lsa_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            rf_cls = LDA()
            rf_cls.fit(xs, ys)
            return rf_cls
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            return classifier.predict(vd)
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        return None
        #return Converter.to_binary

if __name__ == "__main__":

    cl = Codes_ClassifyUsingLDA_WithLSAValues()
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingLDA_WithLSAValues.txt")
    