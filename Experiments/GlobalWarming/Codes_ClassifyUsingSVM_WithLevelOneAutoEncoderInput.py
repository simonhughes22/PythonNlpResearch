from sklearn import svm
from GwExperimentBase import *
import autoencoder

class Codes_ClassifyUsingSVM_WithLevelOneAutoencoderInput(GwExperimentBase):

    def __init__(self, kernel, c):
        self.kernel = kernel
        self.c = c 

    def create_classifier(self, code):
        def svm_create(xs, ys):
            
            if self.kernel == 'linear':
                svm_cls = svm.LinearSVC(C = self.c, dual = True)
            else:
                svm_cls = svm.SVC(kernel = self.kernel, C = self.c)
            svm_cls.fit(xs, ys)
            return svm_cls
        return svm_create
    
    def get_vector_space(self, tokenized_docs):
        xs = autoencoder.load_from_file()        
        return (xs, dict())
    
    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix
    
    def label_mapper(self):    
        return Converter.get_svm_val

if __name__ == "__main__":
    kernel = 'linear'
    C = 5.0
    
    cl = Codes_ClassifyUsingSVM_WithLevelOneAutoencoderInput(kernel, float(C))
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingSVM_WithLevelOneAutoencoderInput_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
