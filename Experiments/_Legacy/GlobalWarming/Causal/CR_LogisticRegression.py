
from sklearn.linear_model import LogisticRegression
import Converter
from GwCrExperimentBase import GwCrExperimentBase

class CR_LogisticRegression(GwCrExperimentBase):

    def __init__(self, C, adjust_class_weight):
        self.C = C        
        self.adjust_class_weight = adjust_class_weight

    def create_classifier(self, code):
        def cls_create(xs, ys):

            if self.adjust_class_weight:
                cls = LogisticRegression('l2', dual=True, C = self.C, class_weight = 'auto')
            else:
                cls = LogisticRegression('l2', dual=True, C = self.C)
            cls.fit(xs, ys)
            return cls
        return cls_create
    
    def classify(self):
        def classify(classifier, vd):
            return classifier.predict(vd)
        return classify

    def get_vector_space(self, tokenized_docs):
        return self.term_freq_vspace(tokenized_docs)

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        #return None
        return Converter.to_binary

if __name__ == "__main__":

    #C = 1.0

    best_f1 = 0
    bestC = 0
    adjust_class_weight = False
    s_cls_wt = ""
    if adjust_class_weight:
        s_cls_wt = "_WeightedByClass"
      
    #best c is 3 so far
    for c in range(3,4,1):
        
        cl = CR_LogisticRegression(float(c), adjust_class_weight)
        (mean_metrics, wt_mean_metrics) = cl.Run("CR_LogisticRegression_C" + str(c) + s_cls_wt + ".txt", min_word_count = 2)
            
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = c
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
