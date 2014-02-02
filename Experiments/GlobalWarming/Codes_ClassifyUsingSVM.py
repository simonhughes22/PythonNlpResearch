from sklearn import svm
from GwExperimentBase import *
from MatrixHelper import unit_vector

class Codes_ClassifyUsingSVM(GwExperimentBase):

    def __init__(self, kernel, c):
        self.kernel = kernel
        self.c = c 

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
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
        """ Returns unit vectors as that is what tf_idf returns
        """
        matrix = self.get_sparse_matrix_data(distance_matrix, id2word)
        return matrix

    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":
    kernel = 'linear'
    C = 5.0
    
    #cl = Codes_ClassifyUsingSVM(kernel, float(C))
    #(mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingSVM_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
    
    bestC = 1.0
    best_f1 = 0   
     
    for C in range(1, 20):
        cl = Codes_ClassifyUsingSVM(kernel, float(C))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingSVM_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = C
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
