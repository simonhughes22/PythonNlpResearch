from sklearn.lda import LDA
from GwExperimentBase import *
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, RFECV
from sklearn.pipeline import Pipeline
from Metrics import rpf1

class Codes_ClassifyUsingLDA_WithTfIdfValues_AndFeatureSelection(GwExperimentBase):

    def __init__(self):
        self.mean = None
        self.sd = None
        pass #no params     

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            def score_fn(expected, actual):
                r,p,f1 = rpf1(expected, actual)
                return 1.0 - f1
            
            clf = LDA()
            selector = RFECV(clf, step=50, cv=3, loss_func=score_fn)
            selector = selector.fit(xs, ys)
            return selector
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

    cl = Codes_ClassifyUsingLDA_WithTfIdfValues_AndFeatureSelection()
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingLDA_WithTfIdfValues_AndFeatureSelection.txt")
    