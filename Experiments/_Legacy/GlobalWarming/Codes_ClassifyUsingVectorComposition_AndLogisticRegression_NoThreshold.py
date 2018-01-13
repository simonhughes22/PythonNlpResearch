from collections import defaultdict

from sklearn.linear_model import LogisticRegression
import numpy as np

from GwExperimentBase import *
from LatentWordVectors import LatentWordVectors
import VectorComposition
from SubSampling import over_sample


class Codes_ClassifyUsingVectorComposition_AndLogisticRegression_NoThreshold(GwExperimentBase):

    def __init__(self, num_topics, func, vector_space_func, use_idf):
        self.num_topics = num_topics
        self.func = func
        self.vector_space_func = vector_space_func
        self.use_idf = use_idf
        
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
            
            classifier = LogisticRegression(penalty='l1', dual=False)
            
            new_xs, new_ys = over_sample(xs, ys)
            classifier.fit(new_xs, new_ys)
            return classifier
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            return classifier.predict(vd)
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix
    
    def label_mapper(self):
        return Converter.to_binary

if __name__ == "__main__":
     
    topics = 130                    #Best = 130
    func = VectorComposition.Mean   #Best = Mean
    stem = True                     #Best = True
    normalize = False               #Best = True (if use_idf = False)
    use_idf = True                  #Best = False (if normalize = True)
    """ TODO - TRY OTHER parm_val VALUES """
    
    vector_space = ("Lsa", lambda tokenized_docs :LatentWordVectors.LsaTfIdfSpace(tokenized_docs, topics, aggregation_method="doc", normalize=normalize))
    #vector_space = ("Lsa", lambda tokenized_docs : LatentWordVectors.LsaTfIdfSpace(tokenized_docs, topics, aggregation_method = "sentence", normalize = normalize))
    
    s_norm = ""
    if normalize:
        s_norm = "_Normalize_"

    s_idf = ""
    if use_idf:
        s_idf = "_Idf"

    if not vector_space[0].startswith("Lsa"):
        topics = 0
  
    if vector_space[0] == "Embeddings":
        stem = False
    
    
    cl = Codes_ClassifyUsingVectorComposition_AndLogisticRegression_NoThreshold(topics, func, vector_space[1], use_idf)
    (mean_metrics, wt_mean_metrics) = cl.Run("_VectorSpace_" + vector_space[0] + "_Dims" + str(topics) + "_" + str(func.func_name) + s_idf + s_norm + '_NoThreshold.txt', stem = stem)
    f1_score = wt_mean_metrics.f1_score
    
    """ TODO IGNORE SENTENCES WITH LESS THAN 3 WORDS!
             - 1. TRY OTHER parm_val VALUES 
             - 2. TRY OTHER Kernels
    """
