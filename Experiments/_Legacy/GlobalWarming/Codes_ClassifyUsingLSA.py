from CosineSimilarity import cosine_similarity
from GwExperimentBase import *
from Lsa import *
from sklearn import svm
import Converter

class Lsa_Classifier(object):
    
    def __init__(self, num_topics):
        self.num_topics = num_topics
    
    def to_lsa_vector(self, tokenized_doc):
        vect = self.lsa.to_lsa_vector(tokenized_doc)
        return self.get_sparse_matrix_data([vect], self.lsa.id2Word)[0]

    def get_sparse_matrix_data(self, distance_matrix, id2word):
        return MatrixHelper.gensim_to_numpy_array(distance_matrix, initial_value = 0)
    
    def train(self, xs, ys):
        td = []
        unique_ys = set()
        for i, x in enumerate(xs):
            y = str(ys[i])
            new_row = [r for r in x]
            new_row.append(y)
            td.append(new_row)
            unique_ys.add(y)
            
        tfidf = TfIdf.TfIdf(td)
        self.lsa = Lsa(tfidf, self.num_topics)
        self.lsa_matrix = self.get_sparse_matrix_data(self.lsa.distance_matrix, self.lsa.id2Word)
        
        self.d_class_vectors = dict()
        
        for y_val in unique_ys:
            self.d_class_vectors[y_val] = self.to_lsa_vector([str(y_val)])
    
    def predict_row(self, vd_x):
        
        vector = self.to_lsa_vector(vd_x)
        d_sims = dict()
        for cls, td_vect in self.d_class_vectors.items():
            sim = cosine_similarity(td_vect, vector)
            d_sims[cls] = sim
        
        best_cls, best_sim = sorted(d_sims.items(), key = lambda (cls, sim) : sim, reverse = True)[0] 
        return int(best_cls)
    
    def predict(self, vd_xs):
        return [ self.predict_row(x) for x in vd_xs ]
    
class Codes_ClassifyUsingLSA(GwExperimentBase):
    
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions

    def get_params(self):
        return "Dimensions: " + str(self.dimensions)
    
    def get_vector_space(self, tokenized_docs):
        return (tokenized_docs, dict())
    
    def create_classifier(self, code):
        def create(xs, ys):
            cls = Lsa_Classifier(self.dimensions)
            cls.train(xs, ys)
            return cls
        return create

    def get_training_data(self, distance_matrix, id2word):
        return distance_matrix       
    
if __name__ == "__main__":

    dims = 300
    
    cl = Codes_ClassifyUsingLSA(dims)
    cl.Run("Codes_ClassifyUsingLSA_Dims_" + str(dims) + ".txt")
    