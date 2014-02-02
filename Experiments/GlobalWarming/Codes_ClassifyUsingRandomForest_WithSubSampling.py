from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from GwExperimentBase import *
from SubSampling import over_sample
from FindThreshold import best_threshold_for_f1, apply_threshold

class Codes_ClassifyUsingRandomForest_WithSubSampling(GwExperimentBase):

    def __init__(self, criterion, num_trees):
        self.criterion = criterion
        self.num_trees = num_trees

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            rf_cls = RandomForestClassifier(n_estimators = self.num_trees, criterion=self.criterion,  n_jobs = 1)
            new_xs, new_ys = over_sample(xs, ys)
            rf_cls.fit(new_xs, new_ys)
            
            """ Use original xs """
            probs = rf_cls.predict_proba(xs)
            self.threshold, self.positive, self.negative = best_threshold_for_f1(probs, 20, ys)

            return rf_cls
        return cls_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)

    def classify(self):
        def classify(classifier, vd):
            return apply_threshold(classifier, vd, self.threshold, self.positive, self.negative)
        return classify

if __name__ == "__main__":
    criterion = 'entropy'
    num_trees = 10
    
    cl = Codes_ClassifyUsingRandomForest_WithSubSampling(criterion, num_trees)
    cl.Run("Codes_ClassifyUsingRandomForest_WithSubSampling_Criterion_" + str(criterion) + "_Trees_" + str(num_trees) + "_WithThresholding.txt")
    
    