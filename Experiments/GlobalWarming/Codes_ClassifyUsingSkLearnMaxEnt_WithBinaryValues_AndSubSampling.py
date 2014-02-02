
from sklearn.linear_model import LogisticRegression
from GwExperimentBase import *
from SubSampling import over_sample

class Codes_ClassifyUsingSkLearnMaxEnt_WithBinaryValues_AndSubSampling(GwExperimentBase):

    def __init__(self, C):
        self.C = C        

    def get_vector_space(self, tokenized_docs):
        return self.term_freq_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            cls = LogisticRegression('l2', True, C = self.C)
            
            new_xs, new_ys = over_sample(xs, ys)
            cls.fit(new_xs, new_ys)
            return cls
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            return classifier.predict(vd) 
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        #return None
        return Converter.to_binary

if __name__ == "__main__":

    #C = 1.0

    best_f1 = 0
    bestC = 0
      
    #best c is 3 so far
    for c in range(3,11,1):
        
        cl = Codes_ClassifyUsingSkLearnMaxEnt_WithBinaryValues_AndSubSampling(float(c))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingSkLearnMaxEnt_WithBinaryValues_AndSubSampling_C" + str(c) + ".txt", min_word_count = 2)
            
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = c
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    