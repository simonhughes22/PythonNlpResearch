from collections import defaultdict

from sklearn import metrics, svm
import numpy as np

from GwExperimentBase import *
import LatentWordVectors
import VectorComposition
from Bagging import Bagger


class Codes_ClassifyUsingVectorComposition_WithBagging(GwExperimentBase):

    def __init__(self, num_topics, func, vector_space_func, parm_val, use_idf):
        self.num_topics = num_topics
        self.func = func
        self.vector_space_func = vector_space_func
        self.parm = parm_val
        self.use_idf = use_idf
        self.selector = None
        
    def get_vector_space(self, tokenized_docs):
        
        lat_vector_model = self.vector_space_func(tokenized_docs)
        collapsed = []

        tally = defaultdict(int)
        for doc in tokenized_docs:
            s = set(doc)
            for item in s:
                tally[item] += 1
        
        """ Set to 1.0 if false """
        self.df = defaultdict(lambda: 1.0)
        vocab_size = len(tally.keys()) * 1.0
        
        if self.use_idf:
            for k,v in tally.items():
                self.df[k] = np.log(v)
    
        for doc in tokenized_docs:
            vectors = []
            for token in doc:
                v = lat_vector_model.project(token)
                if v != None:
                    """ Idf value will be 1.0 if False """
                    vectors.append(np.array(v) * self.num_topics / self.df[token])
                    
            collapse = self.func(vectors)
            collapsed.append(collapse)
        
        print "Constructed Vector Space"
        return (collapsed, dict())
    
    def create_classifier(self, code):
        def cls_create(xs, ys):
            
            def create_svm():
                return svm.SVC(C = self.parm, probability=False)
            
            method = 'mean'
            
            classifier = Bagger(create_svm, bootstraps=5, sample_pct= 0.75, method=method)
            
            classifier.fit(xs, ys)
            vals = classifier.predict(xs)            
            if method == 'mean':
            
                best_threshold = -1
                best_f1 = -1
                
                def create_threshold_fn(threshold):
                    def above_threshold(prob):
                            if prob >= threshold:
                                return 1
                            return -1
                    return above_threshold
                
                for i in range(9):
                    threshold = (i + 1.0) / 5.0 - 1.0
                    new_ys = map(create_threshold_fn(threshold), vals)
                    score = metrics.f1_score(ys[:,0], new_ys)
                    if score > best_f1:
                        best_threshold = threshold
                        best_f1 = score
    
                self.threshold = max(-1, best_threshold)
            else:
                self.threshold = 0.1

            return classifier
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            
            vals = classifier.predict(vd)
            
            def above_threshold(prob):
                if prob >= self.threshold:
                    return 1
                return -1
            
            return map(above_threshold, vals)
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":
     
    topics = 130                    #Best = 130
    func = VectorComposition.CircularConvolution   #Best = Mean
    stem = True                     #Best = True
    normalize = False               #Best = True (if use_idf = False)
    use_idf = True                  #Best = False (if normalize = True)
    """ TODO - TRY OTHER parm_val VALUES """
    
    vector_space = ("Lsa", lambda tokenized_docs : LatentWordVectors.LatentWordVectors.LsaSpace(tokenized_docs, topics, normalize=normalize))
    #vector_space = ("PpmiLsa", lambda tokenized_docs : PpmiWordVectors.PpmiLatentWordVectors(tokenized_docs, topics))
    #vector_space = ("Ppmi", lambda tokenized_docs : PpmiWordVectors.PpmiWordVectors(tokenized_docs))
    #vector_space = ("Embeddings", lambda tokenized_docs : Embeddings(suppress_errors=True)) # Does awfully
    
    s_norm = ""
    if normalize:
        s_norm = "_Normalize_"

    s_idf = ""
    if use_idf:
        s_idf = "_Idf"

    if vector_space[0] != "Lsa":
        topics = 0
    
    if vector_space[0] == "Embeddings":
        stem = False

    best_f1 = -1.0
    s_params = ""
    for parm_val in range(2, 3): #Best C val is 2
        s_params = str(parm_val)

        cl = Codes_ClassifyUsingVectorComposition_WithBagging(topics, func, vector_space[1], (parm_val), use_idf)
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingVectorComposition_VectorSpace_" + vector_space[0] + "_Dims" + str(topics) + str(func.func_name) + s_idf + s_params + s_norm + '.txt', stem = stem)

        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = parm_val
            print "Best parm_val Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best parm_val Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    """ TODO IGNORE SENTENCES WITH LESS THAN 3 WORDS!
             - 1. TRY OTHER parm_val VALUES 
             - 2. TRY OTHER Kernels
    """
    