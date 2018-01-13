from sklearn import svm
from GwLsa import *
from GwExperimentBase import *
from GwCodeTemplates import *
import Converter
from CosineSimilarity import cosine_similarity
import MatrixHelper

class TemplateClassifier(object):
    def __init__(self, lsa, code, templates):
        
        match = filter(lambda tpl: code in tpl[0], templates)
        
        self.lsa = lsa
        distance_matrix = [lsa.project(d) 
                          for c,d in match]
        self.templates = MatrixHelper.gensim_to_numpy_array(distance_matrix, lsa.num_topics)
    
    def get_similarity(self, x):
        max_sim = 0
        for dbnetwork in self.templates:
            sim = cosine_similarity(x, dbnetwork)
            max_sim = max(max_sim, sim)
        return max_sim
    
    def train(self, xs, ys):
        positive_count = len([l for l in ys if l == 1])
        predictions = []
        sims = []
        
        z = zip(xs, ys)
        
        for x,y in z:
            sim = self.get_similarity(x)
                
            predictions.append((y, sim))
            sims.append(sim)
        
        sims = sorted(sims, reverse = True)
        self.threshold = sims[positive_count - 1]
    
    def classify(self, x):        
        sim = self.get_similarity(x)
        if sim >= self.threshold:
            return 1
        else:
            return 0

class Codes_ClassifyUsingLsaSimilarityToTemplates(GwExperimentBase):
    
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.lsa = None
        templates = GwCodeTemplates()
        self.code_templates = zip(templates.codes_per_document, templates.documents)
    
    def get_params(self):
        return "Dimensions: " + str(self.dimensions)
    
    def get_vector_space(self, tokenized_docs):
        if self.lsa == None:
            self.lsa = GwLsa(num_topics = self.dimensions)
        
        distance_matrix = [self.lsa.project(tokenized_doc) 
                  for tokenized_doc in tokenized_docs]
        return (distance_matrix, self.lsa.id2Word)
    
    def create_classifier(self, code):
        def create_classifier(xs, ys):
            clsfr = TemplateClassifier(self.lsa, code, self.code_templates)
            clsfr.train(xs, ys)
            return clsfr
            
        return create_classifier
    
    def classify(self):
        def classify(clsfr, vd):
            return [clsfr.classify(d) for d in vd]
        return classify

    def get_training_data(self, distance_matrix, id2word):
        return self.get_sparse_matrix_data(distance_matrix, id2word)

if __name__ == "__main__":

    dims = 500
    cl = Codes_ClassifyUsingLsaSimilarityToTemplates(dims)
    cl.Run("Codes_ClassfyUsingLSASimilarityToTemplate_Dims_" + str(dims) + ".txt")
    
    