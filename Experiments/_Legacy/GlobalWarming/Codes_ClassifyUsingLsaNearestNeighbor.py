from CosineSimilarity import cosine_similarity
from GwCodeTemplates import *
from GwExperimentBase import *
from GwLsa import *
from collections import defaultdict
from sklearn import svm
from gensim import similarities
import Converter
import MatrixHelper
import numpy as np

class Codes_ClassifyUsingLsaNearestNeighbor(object):
    def __init__(self, lsa, k):
        self.lsa = lsa
        self.k = k
    
    def train(self, xs, ys):
        
        distance_matrix = [[c for c in r]
                  for r in xs]
        
        self.similarity_index = similarities.MatrixSimilarity(distance_matrix)
        
        self.training_data = zip(distance_matrix, ys)
        self.positive_count = len([l for l in ys if l == 1])
        qtr_pos = int(round(self.positive_count / 4.0))
        if self.k > (qtr_pos):
            self.k = qtr_pos
            if self.k < 1:
                self.k = 1
        
        self.total_count = len(xs)
        
        count_per_label = defaultdict(int)
        for x,y in self.training_data:
            count_per_label[y] += 1
        self.total_positive = count_per_label[1]
        self.total_negative = count_per_label[0]

    def classify_best_n(self, vd_x):       
        
        n = self.k
        total_sim = defaultdict(float)
        labelled_sims = []

        row = [(x[0], x[1]) for x in vd_x]
        
        n = min(n, len(self.training_data))        
        sims = enumerate(self.similarity_index[row])
        s_sims = sorted(sims, key = lambda (i,sim): sim, reverse = True)
        top_n = s_sims[:n]

        tally = dict()
        for i,sim in top_n:
            x,y = self.training_data[i]
            if y not in tally:
                tally[y] = sim
            else:
                tally[y] += sim
        
        s_tally = sorted(tally.items(), key = lambda (y, sim): -sim )
        return s_tally[0][0]
        
    def classify_mean_sim(self, vd_x):       
        
        total_sim = defaultdict(float)
        labelled_sims = []

        row = [(x[0], x[1]) for x in vd_x]
        sims = self.similarity_index[row]
        for i, sim in enumerate(sims):
            x, y = self.training_data[i]
            total_sim[y] += sim
            labelled_sims.append((sim, y))
        
        mean_positive = total_sim[1] / self.total_positive
        mean_negative = total_sim[0] / self.total_negative
        
        if mean_positive > mean_negative:
            return 1
        else:
            return 0

class Codes_ClassifyUsingLsaNearestNeighbor(GwExperimentBase):
    
    def __init__(self, num_dimensions, k):
        self.dimensions = num_dimensions
        self.k = k

    def get_params(self):
        return "Dimensions: " + str(self.dimensions)
    
    def get_vector_space(self, tokenized_docs):
        self.lsa = GwLsa(num_topics = self.dimensions)
        distance_matrix = self.lsa.project_matrix(tokenized_docs)
        return (distance_matrix, self.lsa.id2Word)

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix

    def create_classifier(self, code):
        def create_classifier(xs, ys):
            clsfr = NearestNeighborClassifier(self.lsa, self.k)
            clsfr.train(xs, ys)
            return clsfr
            
        return create_classifier
    
    def classify(self):
        def classify_mean_sim(clsfr, vd):
            return [clsfr.classify_mean_sim(d) for d in vd]
        
        def classify_best_n(clsfr, vd):
            return [clsfr.classify_best_n(d) for d in vd]
        
        if self.k <= 0:
            return classify_mean_sim
        
        return classify_best_n

if __name__ == "__main__":

    dims = 300
    k = 1 # k in knn. 0 = mean
    
    for k in range(1,2):
        cl = Codes_ClassifyUsingLsaNearestNeighbor(dims, k)
        cl.Run("Codes_ClassifyUsingLsaNearestNeighbor_Dims_{0}_k_{1}.txt".format(dims, k))
    
    