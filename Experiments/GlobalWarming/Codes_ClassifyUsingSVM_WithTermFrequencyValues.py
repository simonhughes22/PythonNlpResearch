from sklearn import svm
from sklearn.linear_model import LogisticRegression
from GwExperimentBase import *

class Codes_ClassifyUsingSVM_WithTermFrequencyValues(GwExperimentBase):

    def __init__(self, kernel, c):
        self.kernel = kernel
        self.c = c 

    def get_vector_space(self, tokenized_docs):
        return self.term_freq_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def create(xs, ys):
            
            if self.kernel == 'linear':
                svm_cls = svm.LinearSVC(C = self.c, dual = True)
            else:
                svm_cls = svm.SVC(kernel = self.kernel, C = self.c)
            svm_cls.fit(xs, ys)
            return svm_cls
        return create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        return None
        #return Converter.to_binary

if __name__ == "__main__":

    #C = 1.0
    kernel = 'linear'

    best_f1 = 0
    bestC = 0    
    for c in range(5,6,1):
        
        cl = Codes_ClassifyUsingSVM_WithTermFrequencyValues(kernel, float(c))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingSVM_WithTermFrequencyValues_kernel_" + kernel + "_C" + str(c) + ".txt")
            
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = c
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    