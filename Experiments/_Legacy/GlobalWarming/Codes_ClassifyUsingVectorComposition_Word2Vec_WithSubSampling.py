from collections import defaultdict

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from FindThreshold import best_threshold_for_f1, apply_threshold
from GwExperimentBase import *
from SubSampling import over_sample
import VectorComposition


class Codes_ClassifyUsingVectorComposition_Word2Vec_WithSubSampling(GwExperimentBase):

    def __init__(self, num_topics, func, vector_space_func, algo, parm_val, use_idf):
        self.num_topics = num_topics
        self.func = func
        self.vector_space_func = vector_space_func
        self.parm = parm_val
        self.use_idf = use_idf
        self.algo = algo

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
                    vectors.append(np.array(v) / self.df[token])

            collapse = self.func(vectors)
            collapsed.append(collapse)
        
        print "Constructed Vector Space"
        return (collapsed, dict())
    
    def create_classifier(self, code):
        def cls_create(xs, ys):

            #print("Training classifier")
            if algo == "SVM":
                classifier = svm.SVC(C = self.parm, probability=True)
                
            elif algo == "RF":
                classifier = RandomForestClassifier(n_estimators = int(self.parm), criterion='entropy',  n_jobs = 1)

            elif algo == "LogisticRegression":
                classifier = LogisticRegression()
            else:
                raise Exception("Unknown algorithm: " + algo)

            new_xs, new_ys = over_sample(xs, ys)

            classifier.fit(new_xs, new_ys)
            probs = classifier.predict_proba(xs)            
            
            self.threshold, self.positive, self.negative = best_threshold_for_f1(probs, 5, ys)
            # Override threshold
            return classifier
        return cls_create

    def classify(self):
        def classify(classifier, vd):
            #print("Classifying")

            #return classifier.predict(vd)
            return apply_threshold(classifier, vd, self.threshold, self.positive, self.negative)
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix
    
    def label_mapper(self):
        return Converter.get_svm_val

if __name__ == "__main__":

    def Add_UnNormalized(vectors):
        return np.cumsum(vectors, 0)[-1]

    topics = 100                    #
    func = Add_UnNormalized         #
    stem = True                     #
    use_tf  = False                 #
    use_idf = False                 #
    algo = "LogisticRegression"     #
    spelling_correct = True         #

    class GwWord2Vec(object):
        def __init__(self):
            import GwData
            import WordTokenizer
            from py_word2vec import Word2Vec

            data = GwData.GwData()
            # Ensure we train on all words here (as a sequence model), stem matches
            # above setting, and we do NOT remove stop words (breaks sequencing)
            # spelling correct must match below
            tokenized_docs = WordTokenizer.tokenize(data.documents,
                        min_word_count=1, stem = stem,
                        remove_stop_words=False, spelling_correct=spelling_correct,
                        number_fn=NumberStrategy.collapse_num)

            self.wd2vec = Word2Vec(tokenized_docs, topics, min_count=2)

        def project(self, word):
            if word not in self.wd2vec.vocab:
                raise Exception("Missing word: " + word)
            return self.wd2vec[word]

    """ TODO - TRY OTHER parm_val VALUES """
    gwWord2Vec = GwWord2Vec()
    vector_space = ("Word2Vec", lambda tokenized_docs : gwWord2Vec)
    
    s_idf = ""
    if use_idf:
        s_idf = "_Idf"

    if vector_space[0] == "Embeddings":
        stem = False

    s_params = ""
    if algo == "SVM":
        s_params = "_C_" 
    elif algo =="RF":
        s_params = "_Trees_"
    
    best_f1 = -1.0
    SC = ""
    if spelling_correct:
        SC = "_SC"

    for parm_val in range(2, 3): #Best C val is 2
        if algo == "SVM":
            s_params += str(parm_val)

        cl = Codes_ClassifyUsingVectorComposition_Word2Vec_WithSubSampling(topics, func, vector_space[1], algo, (parm_val), use_tf, use_idf)

        """ Regular name too long """

        #(mean_metrics, wt_mean_metrics) = cl.Run('test_run_50.txt',stem=stem, spelling_correct=spelling_correct, one_code="50", one_fold=True)

        (mean_metrics, wt_mean_metrics) = cl.Run(vector_space[0] + "_Dims" + str(topics)
                                                 + "_" + algo + "_" + str(func.func_name) + s_idf + s_params  + SC + '.txt',
                                                 stem = stem, spelling_correct=spelling_correct)

        f1_score = wt_mean_metrics.f1_score
        if f1_score > best_f1:
            best_f1 = f1_score
            bestC = parm_val
            print "Best parm_val Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    print "Best parm_val Value: {0} with F1: {1}".format(str(bestC), best_f1)
    
    """

    """
    