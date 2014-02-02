from sklearn import svm
from GwExperimentBase import *
from StackedExperimentRunner import StackedExperimentRunner

class Stacked_Codes_ClassifyUsingSVM(GwExperimentBase, StackedExperimentRunner):

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
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":

    kernel = 'linear'
    C = 5.0
    
    #cl = Codes_ClassifyUsingSVM(kernel, float(C))
    #(mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingSVM_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
    
    bestC = 1.0
    best_f1 = 0   
     
    #for C in range(1, 20):
    cl = Stacked_Codes_ClassifyUsingSVM(kernel, float(C))
    cl.RunStacked("Stacked_Codes_ClassfyUsingSVM_kernel_{0}_c_{1}.txt".format(kernel, str(C)), cv_folds=10, layers=4)
