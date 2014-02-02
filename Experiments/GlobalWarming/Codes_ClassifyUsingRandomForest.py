from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from GwExperimentBase import *

class Codes_ClassifyUsingRandomForest(GwExperimentBase):

    def __init__(self, criterion, num_trees):
        self.criterion = criterion
        self.num_trees = num_trees

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            rf_cls = RandomForestClassifier(n_estimators = self.num_trees, criterion=self.criterion,  n_jobs = -1)
            rf_cls.fit(xs, ys)
            return rf_cls
        return cls_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)

if __name__ == "__main__":
    criterion = 'entropy'
    num_trees = 100
    
    cl = Codes_ClassifyUsingRandomForest(criterion, num_trees)
    cl.Run("Codes_ClassifyUsingRandomForest_Criterion_" + str(criterion) + "_Trees_" + str(num_trees) + ".txt")
    
    