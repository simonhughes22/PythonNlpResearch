import datetime as dt

from sklearn import svm
import numpy as np

from GwExperimentBase import *
from WordVectors.Embeddings import *
from deepnet import *


class Codes_ClassifyUsingRRBM(GwExperimentBase):

    def __init__(self, hidden_nodes, stem):
        self.hiden_nodes = hidden_nodes
        self.embeddings = Embeddings(0.95, 0.05, stem = stem)
        self.rbm = None
        self.d_memoized = None
        self.code = ""
        self.intermediate_xs = None
        self.intermediate_ys = None

    def get_vector_space(self, tokenized_docs):
        e =  [self.embeddings.project(token) 
            for token in tokenized_docs]

        #enorm = [[self.__normalize_length__(i) for i in r] for r in e]
        return (e, dict())
        
        """"e = []
        for doc in tokenized_docs:
            emb = self.embeddings.project(doc)
            row = []
            for etoken in emb:
                row.append(self.__normalize_length__(np.array(etoken)))
            e.append(row)
        
        return (e, dict())
        """
        
    def __normalize__(self, x, mult):
        xn = x * mult
        return min(1.0, xn)
    
    def __normalize_length__(self, vect):
        return vect.as_numpy_array() 
        
        """total = sum(vect)
        mult = 10.0 / total
        return [self.__normalize__(v, mult) for v in vect]
        """
        
    def __get_input__(self, x, data):
        for i, v in enumerate(x):
            if i == 0:
                continue
            a = x[i-1]
            b = x[i]
            data.append(np.array(a + b))
        return data
      
    def project(self, x, errors = None, to_classify = None, new_inputs = None):
        #get pairwise combinations
        
        if len(x) == 0:
            return []
        
        if len(x) == 1:
            return x[0]
        
        if self.d_memoized != None:
            key = str(x)
            if key in self.d_memoized:
                return self.d_memoized[key]
            
        activations = self.__get_activations__(x, errors, to_classify, new_inputs)
        if self.d_memoized != None:
            self.d_memoized[key] = activations
        return activations
        
    def __get_activations__(self, x, errors, to_classify, new_inputs):
        row = x[:]
        inputs = self.__get_input__(row, [])
        if to_classify != None:
            to_classify.extend(inputs)
        
        activations = self.rbm.prop_up(inputs)
        if len(activations) == 1:
            return self.__normalize_length__(activations[0])
        
        out = self.rbm.prop_down(activations).as_numpy_array()
        diff = np.abs(inputs - out)
        
        lowest_error_ix = 0
        lowest_error = 1000000
        
        for i,err in enumerate(diff):
            mae = np.mean(err)
            if mae < lowest_error:
                lowest_error = mae
                lowest_error_ix = i
        
        if errors != None:
            errors.append(lowest_error)
            
        new_input = self.__normalize_length__(activations[lowest_error_ix])
       
        while (True):
            
            len_inputs = len(inputs)
            if new_inputs != None:
                new_inputs.append(new_input)

            if len_inputs == 2:
                return new_input
            
            # if first replace first pair only
            row = row[:lowest_error_ix] + [new_input.tolist()] + row[lowest_error_ix + 2:]
            inputs = self.__get_input__(row, [])

            if lowest_error_ix == 0:
                new_pairs = [inputs[0]]
            elif lowest_error_ix == len(inputs):
                new_pairs = [inputs[-1]]
            else:
                new_pairs = inputs[lowest_error_ix -1:lowest_error_ix + 1]
            
            assert (len_inputs - 1) == len(inputs)
            assert len(new_pairs) in [1,2]
            # end if statement to make inputs
  
            if to_classify != None:
                to_classify.extend(new_pairs)
  
            activations = self.rbm.prop_up(inputs)
            out = self.rbm.prop_down(activations).as_numpy_array()
            diff = np.abs(inputs - out)
            
            lowest_error_ix = 0
            lowest_error = 1000000
            
            for i,err in enumerate(diff):
                mae = np.mean(err)
                if mae < lowest_error:
                    lowest_error = mae
                    lowest_error_ix = i
            
            if errors != None:
                errors.append(lowest_error)

            new_input = activations[lowest_error_ix].as_numpy_array()           

     
    def create_classifier(self, code):
        def create(xs, ys):
            
            if self.code != code:
                self.code = code
                # Reset memoizer otherwise will simply memoize results
                self.d_memoized = None
                
                initial_pass = 10
                training = 10
                
                self.rbm = RBM(100, 50)
                
                if initial_pass > 0:
                    # pre train on word vectors only initially
                    td_first_pass_neg = []
                    td_first_pass_pos = []
                    for i,x in enumerate(xs):
                        if ys[i] == 1:
                            self.__get_input__(x, td_first_pass_pos)
                        else:
                            self.__get_input__(x, td_first_pass_neg)
                    
                    self.rbm.train(td_first_pass_pos, initial_pass, 0.05, early_stop = False)
                    self.rbm.train(td_first_pass_neg, initial_pass, 0.05, early_stop = False)
                    
                    td_first_pass_neg = None
                    td_first_pass_pos = None
                    
                intermediate_xs = []
                intermediate_ys = []
                
                for epoch in range(training):

                    epoch_start = dt.datetime.now()
                    
                    all_errors = []
                    to_classify_neg = []
                    to_classify_pos = []
                    
                    last_epoch = epoch == (training -1)
                    
                    for i,x in enumerate(xs):
                        if len(x) == 0 or len(x[0]) == 0:
                            assert 1 == 2
                    
                        to_classify = []
                        y = ys[i]
                        
                        if last_epoch:
                            new_inputs = []
                            xout = self.project(x, all_errors, to_classify, new_inputs)
                            
                            intermediate_xs.extend(new_inputs)
                            intermediate_ys.extend([y for x in new_inputs])
                        else:
                            xout = self.project(x, all_errors, to_classify)
                        
                        assert len(xout) == 50
                        
                        if y == 1:
                            to_classify_pos.extend(to_classify)                            
                        else:
                            to_classify_neg.extend(to_classify)
                                                    
                        if len(to_classify_neg) + len(to_classify_pos) >= 1000:
                            
                            self.rbm.train(to_classify_pos, 3, 0.15, early_stop = False, verbose = False)
                            self.rbm.train(to_classify_neg, 3, 0.15, early_stop = False, verbose = False)
                            to_classify_pos = []
                            to_classify_neg = []
                                                
                        if i % 250 == 0:
                            x_min = min(xout)
                            x_max = max(xout)
                            ix_min = 0
                            ix_max = 0
                            for ix, xo in enumerate(xout):
                                if xo == x_min:
                                    ix_min = ix
                                if xo == x_max:
                                    ix_max = ix
                            print "{0}: Min {1} ix Min: {2} Max {3} ix Max: {4} Sum {5}".format(str(i), str(x_min), str(ix_min), str(x_max), str(ix_max), str(sum(xout)))
                            
                        # repeat until data of length 1
                    if len(to_classify_neg) + len(to_classify_pos) >= 1000:
                            self.rbm.train(to_classify_pos, 3, 0.15, early_stop = False, verbose = False)
                            self.rbm.train(to_classify_neg, 3, 0.15, early_stop = False, verbose = False)
                     
                    print "\nMAE for epoch: " + str(epoch) + " is " + str(np.mean(all_errors))
                    duration = dt.datetime.now() - epoch_start
                    print "Epoch lasted: " + str(duration.total_seconds()) + " seconds"
            
                # once trained, memoize
                self.memoize()
                self.intermediate_xs = intermediate_xs
                self.intermediate_ys = intermediate_ys
                
            data = [self.project(x) for x in xs]
            
            all_data = data[:]
            all_data.extend(self.intermediate_xs)
                
            all_ys = ys.tolist()
            all_ys.extend(self.intermediate_ys)

            assert (len(all_data) == len(all_ys))
            
            #classifier = RandomForestClassifier(n_estimators = 100, criterion='entropy',  n_jobs = 8)
            #classifier = LogisticRegression('l2', True, C = self.C)
            #classifier = svm.LinearSVC(C = 5, class_weight = 'auto')
            #classifier = svm.SVC()
            
            #classifier = svm.NuSVC()
            classifier = svm.SVC(C = 5.0, probability=True)
            classifier.fit(all_data, all_ys)
            
            probs = classifier.predict_proba(data)
            pclassifier = svm.SVC(C = 5.0)
            pclassifier.fit(probs, ys)
            self.pclassifier = pclassifier
            
            print type(classifier), type(pclassifier)
            return classifier
        return create
    
    def classify(self):
        def classify(classifier, vd):
            data = [self.project(x) for x in vd]
            probs = classifier.predict_proba(data)            
            return self.pclassifier.predict(probs)              
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix
    
    def label_mapper(self):
        return Converter.get_svm_val
    
    def memoize(self):
        self.d_memoized = dict()

if __name__ == "__main__":
    
    hidden_nodes = int(50) 
    stem = False
    
    cl = Codes_ClassifyUsingRRBM(hidden_nodes, stem)
    (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassfyUsingRRBM_with_RF_with_{0}_nodes_Remove_Bad_Words.txt".format(str(hidden_nodes)), 
                                             min_word_count = 5, stem = stem, remove_stop_words = True, 
                                             one_code = "20")
    f1_score = wt_mean_metrics.f1_score
    
    