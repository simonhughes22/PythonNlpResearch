from sklearn import svm
from GwLsa import *
from GwExperimentBase import *
import Converter

class Codes_ClassifyUsingSVM_WithLSA(GwExperimentBase):
    
    def __init__(self, num_dimensions, c):
        self.dimensions = num_dimensions
        self.C = c

    def get_params(self):
        return "Dimensions: " + str(self.dimensions) + " C: " + str(self.C)
    
    def get_vector_space(self, tokenized_docs):
        lsa = GwLsa(num_topics = self.dimensions)
        
        distance_matrix = [lsa.project(tokenized_doc) 
                  for tokenized_doc in tokenized_docs]
        return (distance_matrix, lsa.id2Word)
    
    def create_classifier(self, code):
        def svm_create(xs, ys):
            svm_cls = svm.LinearSVC(C = self.C, dual = True)
            svm_cls.fit(xs, ys)
            return svm_cls
        return svm_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":

    dims = 300
        
    for c in range(5, 6):
        cl = Codes_ClassifyUsingSVM_WithLSA(dims, float(c))
        cl.Run("Codes_ClassfyUsingSVM_WithLSA_Dims_" + str(dims) + "_C_" + str(c) + ".txt")