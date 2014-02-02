from sklearn import svm, preprocessing
from GwExperimentBase import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV, SelectPercentile, f_classif, chi2
from sklearn.metrics import f1_score

class Codes_ClassifyUsingSVM_Normalized_FeatureSelection(GwExperimentBase):

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
            
            #def score(y_true, y_pred):
            #   return 1.0 - f1_score(y_true, y_pred)
            
            #rfecv = RFECV(estimator=svm_cls, step=50, cv=StratifiedKFold(ys, 2))
            #rfecv.fit(xs, ys)
            
            """ Feature Selection """
            self.selector = svm.LinearSVC(C = 1, dual = False, penalty="l1")
            new_xs = self.selector.fit_transform(xs, ys)
            
            """ Training """
            svm_cls.fit(new_xs, ys)
            
            return svm_cls
        return svm_create

    def classify(self):
        def classify(classifier, vd):
            new_xs =self.selector.transform(vd)
            return classifier.predict(new_xs) 
        return classify

    def get_training_data(self, distance_matrix, id2word):
        print "Desparsifying Matrix"
        m =self.get_sparse_matrix_data(distance_matrix, id2word)
        if self.kernel == "rbf":
            """ Must be scaled """
            print "Scaling Matrix"
            return preprocessing.scale(m)
        return m

    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":
    kernel = 'linear'
    C = 5.0
    
    #cl = Codes_ClassifyUsingSVM(kernel, float(C))
    #(mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingSVM_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
    
    bestC = 1.0
    best_f1 = 0   
     
    for C in range(2, 3):
        cl = Codes_ClassifyUsingSVM_Normalized_FeatureSelection(kernel, float(C))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingSVM_Normalized_FeatureSelection_kernel_{0}_c_{1}.txt".format(kernel, str(C)))
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = C
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    