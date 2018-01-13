from sklearn import svm
from GwExperimentBase import *

class Codes_ClassifyUsingSVM_WithLSA_UsingOnlyEssays(GwExperimentBase):

    def __init__(self, kernel, c, num_topics):
        self.kernel = kernel
        self.c = c
        self.num_topics = num_topics

    def get_vector_space(self, tokenized_docs):
        return self.lsa_vspace(tokenized_docs)

    def create_classifier(self, code):
        def svm_create(xs, ys):
            
            if self.kernel == 'linear':
                svm_cls = svm.LinearSVC(C = self.c, dual = True)
            else:
                svm_cls = svm.SVC(kernel = self.kernel, C = self.c)
            svm_cls.fit(xs, ys)
            return svm_cls
        return svm_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":
    C = 5.0
    num_topics = 100
    cl = Codes_ClassifyUsingSVM_WithLSA_UsingOnlyEssays('linear', float(C), num_topics)
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingSVM_WithLSA_UsingOnlyEssays_Dims_{0}_c_{0}.txt".format(str(num_topics), str(C)))
    
    """
    bestC = 1.0
    best_f1 = 0   
     
    for C in range(1, 100):
        cl = Codes_ClassifyUsingSVM_WithLSA_UsingOnlyEssays(kernel, float(C))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingSVM_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = C
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    """
