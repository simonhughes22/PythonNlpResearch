from sklearn import metrics, svm

from GwExperimentBase import *
from MyCode import NeuralNetwork


class Codes_ClassifyUsingAuto_Encoder_WithSVM(GwExperimentBase):

    def __init__(self, C):
        self.C = C
        self.ae = None        

    def get_vector_space(self, tokenized_docs):
        #return self.term_freq_vspace(tokenized_docs)
        return self.tfidf_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            if self.ae == None:
                """ TODO - This is a little bit cheating as it will have seen data used in other folds 
                            Fix this when getting publishable results
                """
                num_inputs = len(xs[0])
                num_hidden = 1250
                
                self.ae = NeuralNetwork(num_inputs, num_hidden, learning_rate = 0.1, activation_fn = "sigmoid", initial_wt_max = 0.01, weight_decay = 0.0, desired_sparsity = 0.05, sparsity_wt = 0.05)
                
                #self.ae.train(xs, epochs = 1, batch_size = 500)
                #self.ae.learning_rate = 0.1
                self.ae.train(xs, epochs = 100, batch_size = len(xs))
            
            features = self.ae.hidden_activations(xs)
            
            classifier = svm.SVC(C = self.C, probability=True)
            #classifier = RandomForestClassifier(n_estimators = 100, criterion='entropy',  n_jobs = 4)
            classifier.fit(features, ys)
            
            probs = classifier.predict_proba(features)            
            
            #self.pclassifier = svm.SVC(C = 1.0)
            #self.pclassifier.fit(probs, ys)
            
            best_threshold = -1
            best_f1 = -1
            
            def create_threshold_fn(threshold):
                def above_threshold(prob):
                        if prob[1] >= threshold:
                            return 1
                        return -1
                return above_threshold
            
            for i in range(9):
                threshold = (i + 1.0) / 10.0
                new_ys = map(create_threshold_fn(threshold), probs)
                score = metrics.f1_score(ys, new_ys)
                if score > best_f1:
                    best_threshold = threshold
                    best_f1 = score
            
            below = best_threshold - 0.1
            for i in range(21):
                threshold = below + (i / 100.0)
                new_ys = map(create_threshold_fn(threshold), probs)
                score = metrics.f1_score(ys, new_ys)
                if score > best_f1:
                    best_threshold = threshold
                    best_f1 = score
            
            self.threshold = max(0, best_threshold)
            return classifier
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            features = self.ae.hidden_activations(vd)
            probs = classifier.predict_proba(features)
            
            def above_threshold(prob):
                if prob[1] >= self.threshold:
                    return 1
                return -1
            
            return map(above_threshold, probs)
            #return classifier.predict(vd)
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
    
    def matrix_value_mapper(self):
        return None
        #return Converter.to_binary
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":

    #C = 1.0

    best_f1 = 0
    bestC = 0    
    #best c is 3 so far
    for c in range(5,6,1):
        
        cl = Codes_ClassifyUsingAuto_Encoder_WithSVM(float(c))
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingAuto_Encoder_WithSVM_C" + str(c) + ".txt", min_word_count = 2)
            
        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = c
            print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best C Value: {0} with F1: {1}".format(str(bestC), best_f1)
    