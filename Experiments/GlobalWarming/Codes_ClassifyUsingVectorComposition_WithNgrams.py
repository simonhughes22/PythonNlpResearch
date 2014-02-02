from FindThreshold import best_threshold_for_f1, apply_threshold
from GwExperimentBase import *
from collections import defaultdict
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from LatentWordVectors import LatentWordVectors as lwv

from NgramGenerator import compute_ngrams
from SkipGramGenerator import compute_skip_grams

import VectorComposition
import numpy as np


class Codes_ClassifyUsingVectorComposition_WithNGrams(GwExperimentBase):

    def __init__(self, num_topics, func, vector_space_func, algo, parm_val, use_idf):
        self.num_topics = num_topics
        self.func = func
        self.vector_space_func = vector_space_func
        self.parm = parm_val
        self.use_idf = use_idf
        self.algo = algo
        self.selector = None

    def get_vector_space(self, tokenized_docs):
        
        def collapse(tag, ngram):
            return tag + ":" + "-".join(ngram)

        collapse_ngram = lambda ngram: collapse("ng", ngram)
        collapse_skip_gram = lambda ngram: collapse("sg", ngram)

        data = []
        df_tally = defaultdict(int)

        for doc in tokenized_docs:
            ngrams = compute_ngrams(doc, 2, 2)
            #skip_grams = compute_skip_grams(doc, 5)
            skip_grams = []

            example = doc + map(collapse_ngram, ngrams) + map(collapse_skip_gram, skip_grams)
            data.append(example)
            # compute doc freq
            s = set(example)
            for item in s:
                df_tally[item] += 1

        # Remove low ngram counts
        processed_data = []
        for example in data:
            row = []
            for term in example:
                if df_tally[term] >= self.min_word_count:
                    row.append(term)
            processed_data.append(row)
        del data # prevent bugs due to later access

        lat_vector_model = self.vector_space_func(processed_data)

        """ Set to 1.0 if false """
        self.df = defaultdict(lambda: 1.0)

        if self.use_idf:
            for k,v in df_tally.items():
                self.df[k] = np.log(v + 1)

        collapsed = []
        for example in processed_data:
            vectors = []
            for token in example:
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
            #
            #classifier = LDA()
            
            new_xs = xs
            
            """
            positive_count = len([y for y in ys if y > 0])
            if positive_count >= 20:
            
                #self.selector = svm.LinearSVC(C = 1, dual = False, penalty="l1")
                self.selector = LDA()
                new_xs = self.selector.fit_transform(xs, ys)
            else:
                self.selector = None
            """
            
            classifier.fit(new_xs, ys)
            probs = classifier.predict_proba(new_xs)            
            
            #self.pclassifier = svm.SVC(parm_val = 1.0)
            #self.pclassifier.fit(probs, ys)
            
            self.threshold, self.positive, self.negative = best_threshold_for_f1(probs, 20, ys)
            return classifier
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            if self.selector != None:
                new_xs = self.selector.transform(vd)
            else:
                new_xs = vd
            
            return apply_threshold(classifier, new_xs, self.threshold, self.positive, self.negative)
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
    min_df = 5                      #Min DF for features
    """ TODO - TRY OTHER parm_val VALUES """

    vector_space = ("Lsa", lambda tokenized_docs : lwv.LsaTfIdfSpace(tokenized_docs, topics, normalize=normalize))
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
    
    if algo == "SVM":
        s_params = "_C_" 
    elif algo =="RF":
        s_params = "_Trees_"
    
    best_f1 = -1.0
    for parm_val in range(2, 3): #Best C val is 2
        if algo == "SVM":
            s_params += str(parm_val)

        cl = Codes_ClassifyUsingVectorComposition_WithNGrams(topics, func, vector_space[1], algo, (parm_val), use_idf)
        (mean_metrics, wt_mean_metrics) = cl.Run("BigramsOnly_VSpace_" + vector_space[0] + "_Dims" + str(topics) + "_" + algo + "_" + str(func.func_name) + s_idf + s_params + s_norm + '.txt', stem = stem)

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
    