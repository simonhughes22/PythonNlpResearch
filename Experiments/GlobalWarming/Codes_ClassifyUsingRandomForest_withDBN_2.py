from sklearn.ensemble import RandomForestClassifier

from GwExperimentBase import *
from deepnet import *

class Codes_ClassifyUsingRandomForest_withDBN(GwExperimentBase):
    sharedDbn = None

    def __init__(self, criterion, num_trees):
        self.criterion = criterion
        self.num_trees = num_trees
        self.dbn = Codes_ClassifyUsingRandomForest_withDBN.sharedDbn
        
    def get_vector_space(self, tokenized_docs):
        return self.term_freq_vspace(tokenized_docs)
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            if self.dbn == None:
              
                dbnetwork = DeepNet([xs.shape[1], 300, 150, 50], ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'], sample = False)
                dbnetwork.train(xs, [500, 500, 500], [0.25, 0.05, 0.05], early_stop=False)
                out = dbnetwork.run_through_network(xs)
              
                # Store for other instances
                Codes_ClassifyUsingRandomForest_withDBN.sharedDbn = dbnetwork
                self.dbn = dbnetwork
            
            rf_cls = RandomForestClassifier(n_estimators = self.num_trees, criterion=self.criterion,  n_jobs = 8)
          
            data = self.dbn.run_through_network(xs)
            rf_cls.fit(data, ys)
            return rf_cls
        return cls_create
        
    def classify(self):
        def classify(classifier, vd):
            data = self.dbn.run_through_network(vd)
            return classifier.predict(data) 
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)
  
    def matrix_value_mapper(self):
        #return None
        return Converter.to_binary

if __name__ == "__main__":
    
    criterion = 'entropy'
    num_trees = 100
    
    cl = Codes_ClassifyUsingRandomForest_withDBN(criterion, num_trees)
    cl.Run("Codes_ClassifyUsingRandomForest_withDBN_Criterion_" + str(criterion) + "_Trees_" + str(num_trees) + ".txt")
    