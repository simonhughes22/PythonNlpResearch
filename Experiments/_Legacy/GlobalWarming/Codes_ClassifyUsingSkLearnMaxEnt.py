from sklearn.linear_model import LogisticRegression
from GwExperimentBase import *

class Codes_ClassifyUsingSkLearnMaxEnt(GwExperimentBase):

    def __init__(self, C):
        self.C = C        

    def get_vector_space(self, tokenized_docs):
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            rf_cls = LogisticRegression('l2', True, C = self.C)
            rf_cls.fit(xs, ys)
            return rf_cls
        return cls_create

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)

if __name__ == "__main__":

    #C = 1.0

    best_f1 = 0
    bestC = 0    
    for c in range(15,55,5):
        
        cl = Codes_ClassifyUsingSkLearnMaxEnt(float(c))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingSkLearnMaxEnt_C" + str(c) + ".txt")
            
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = c
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    