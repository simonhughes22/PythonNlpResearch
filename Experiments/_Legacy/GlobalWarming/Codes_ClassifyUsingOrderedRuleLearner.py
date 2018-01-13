from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from GwExperimentBase import *
from OrderedRuleLearner import *

class Codes_ClassifyUsingOrderedRuleLearner(GwExperimentBase):

    def __init__(self):
        pass

    def create_classifier(self, code):
        def cls_create(xs, ys):

            rf_cls = OrderedRuleLearner()
            rf_cls.fit(xs, ys)
            return rf_cls
        return cls_create
    
    def get_vector_space(self, tokenized_docs):
        return (tokenized_docs, None)

    def get_training_data(self, tokenized_docs, id2word):
        return tokenized_docs

if __name__ == "__main__":

    criterion = 'entropy'
    cl = Codes_ClassifyUsingOrderedRuleLearner()
    cl.Run("Codes_ClassifyUsingOrderedRuleLearner.txt", spelling_correct=False)