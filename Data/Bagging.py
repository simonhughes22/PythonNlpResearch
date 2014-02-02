
from collections import defaultdict
from sklearn.cross_validation import Bootstrap
import numpy as np

class Bagger(object):
    '''
    Wraps any estimator and create a bagger out of it
    '''

    def __init__(self, create_estimator_fn, bootstraps, sample_pct , method = 'vote'):
        self.create_estimator_fn = create_estimator_fn
        self.bootstraps = bootstraps
        self.sample_pct = float(sample_pct)
        self.estimators = None
        """ Use majority voting or average? """
        if method == 'vote':
            assert bootstraps % 2 == 1, "Must be an odd number for voting" 
        self.method = method
    
    def fit(self, x, y):
        
        x = np.array(x)
        
        indices = range(len(x))
        num_per_sample = int(self.sample_pct * len(x))
        self.estimators = []
        
        for i in range(self.bootstraps):
            
            ixs = []
            bootstrap_y = [0]
            
            while min(bootstrap_y) == max(bootstrap_y):
                np.random.shuffle(indices)
                ixs = indices[:num_per_sample]
                bootstrap_y = y[ixs]
                    
            bootstrap_x = x[ixs]
            
            assert len(bootstrap_x) == len(bootstrap_y)
            
            estimator = self.create_estimator_fn()
            estimator.fit(bootstrap_x, bootstrap_y)
            self.estimators.append(estimator)
            
    def __mean__(self, x):
        
        predictions = []
        for est in self.estimators:
            pred = est.predict(x)
            predictions.append(pred)
            
        return np.mean(np.array(predictions).T, 1).tolist()

    def __majority__(self, x):
        
        all_predictions = []
        for est in self.estimators:
            pred = est.predict(x)
            all_predictions.append(pred)
        
        predictions = []
        for i in range(len(x)):
            tally = defaultdict(int)
            for j in range(self.bootstraps):
                val = all_predictions[j][i]
                tally[val] += 1
                
            key, val = sorted(tally.items(), key = lambda (k,v): v)[-1]
            predictions.append(key)
            
        return predictions
    
    def __max__(self, x):
        
        predictions = []
        for est in self.estimators:
            pred = est.predict(x)
            predictions.append(pred)
            
        return np.max(np.array(predictions).T, 1).tolist()
        
    def predict(self, x):
        if self.method == 'vote':
            return self.__majority__(x)
        elif self.method == 'mean' or self.method == "average":
            return self.__mean__(x)
        elif self.method == 'max':
            return self.__max__(x)
        else:
            raise Exception("Method not implemented")