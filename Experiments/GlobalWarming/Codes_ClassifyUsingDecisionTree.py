from GwExperimentBase import *
from sklearn.tree import DecisionTreeClassifier

class Codes_ClassifyUsingDecisionTree(GwExperimentBase):

    def __init__(self, criterion):
        self.criterion = criterion

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            dt = DecisionTreeClassifier(criterion=self.criterion, max_depth=50, min_samples_leaf=2)

            dt.fit(xs, ys)
            return dt
        return cls_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)

if __name__ == "__main__":
    criterion = 'entropy'

    cl = Codes_ClassifyUsingDecisionTree(criterion)
    cl.Run(Codes_ClassifyUsingDecisionTree.__name__ + "_" + str(criterion) +  ".txt")