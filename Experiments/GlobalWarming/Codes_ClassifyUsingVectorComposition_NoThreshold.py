from collections import defaultdict

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from GwExperimentBase import *
import LatentWordVectors
import VectorComposition


class Codes_ClassifyUsingVectorComposition_NoThreshold(GwExperimentBase):

    def __init__(self, num_topics, func, vector_space_func, algo, parm_val, use_idf):
        self.num_topics = num_topics
        self.func = func
        self.vector_space_func = vector_space_func
        self.parm = parm_val
        self.use_idf = use_idf
        self.algo = algo
        self.selector = None
        
    def get_vector_space(self, tokenized_docs):
        
        #fname = "parm_val:\Users\simon.hughes\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\PyDevNLPLibrary\src\Data\GlobalWarming\\lsa_add_embeddings_50.txt"
        #collapsed = CsvUtils.vectors_from_file(fname) * 100.0
        
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
            
            if algo == "SVM":
                classifier = svm.SVC(C = self.parm, probability=True)
                
            elif algo == "RF":
                classifier = RandomForestClassifier(n_estimators = int(self.parm), criterion='entropy',  n_jobs = 1)
            new_xs = xs
            classifier.fit(new_xs, ys)
            return classifier
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            if self.selector != None:
                new_xs = self.selector.transform(vd)
            else:
                new_xs = vd
            return classifier.predict(vd)
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":
     
    topics = 130                    #Best = 130
    func = VectorComposition.Mean   #Best = Mean
    stem = True                     #Best = True
    normalize = False               #Best = True (if use_idf = False)
    use_idf = True                  #Best = False (if normalize = True)
    algo = "SVM"                    #Best is SVM
    """ TODO - TRY OTHER parm_val VALUES """
    
    vector_space = ("Lsa", lambda tokenized_docs : LatentWordVectors.LatentWordVectors.LsaTfIdfSpace(tokenized_docs, topics, normalize=normalize))
     
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
    
    if algo == "SVM":
        s_params = "_C_" 
    elif algo =="RF":
        s_params = "_Trees_"
    
    best_f1 = -1.0
    for parm_val in range(2, 3): #Best C val is 2
        if algo == "SVM":
            s_params += str(parm_val)

        cl = Codes_ClassifyUsingVectorComposition_NoThreshold(topics, func, vector_space[1], algo, (parm_val), use_idf)
        (mean_metrics, wt_mean_metrics) = cl.Run("Codes_ClassifyUsingVectorComposition_NoThreshold_VectorSpace_" + vector_space[0] + "_Dims" + str(topics) + "_" + algo + "_" + str(func.func_name) + s_idf + s_params + s_norm + '.txt', stem = stem)

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
    