from sklearn.qda import QDA
from GwExperimentBase import *
import numpy as np

class Codes_ClassifyUsingQDA_WithTfIdfValues(GwExperimentBase):

    def __init__(self):
        pass #no params     

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            rf_cls = QDA()
            rf_cls.fit(xs, ys)
            return rf_cls
        return cls_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        return None
        #return Converter.to_binary

if __name__ == "__main__":

    cl = Codes_ClassifyUsingQDA_WithTfIdfValues()
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingQDA_WithTfIdfValues.txt")
    