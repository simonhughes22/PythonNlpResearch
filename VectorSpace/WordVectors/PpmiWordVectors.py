from MatrixHelper import gensim_to_numpy_array
from VectorSpaceUtils import compute_id2_word
from gensim.interfaces import CorpusABC
from gensim.models import LsiModel
from itertools import izip, imap
import Ppmi
import ProjectionABC

class PpmiWordVectors(ProjectionABC.ProjectionABC, CorpusABC):
    
    def __init__(self, tokenized_docs):
        ppmis = Ppmi.Ppmi(tokenized_docs)
        self.id2word, word2id = compute_id2_word(tokenized_docs)
        self.num_terms = len(word2id.keys())
        
        def vect_len(vector):
            return sum(v**2.0 for w,v in vector) ** 0.5
        
        vect_lens = map(vect_len, ppmis.values())
        def get_row((vect, len)):
            return map(lambda (k,v): (word2id[k] , v / len), vect)
        
        self.rows = map(get_row, izip(ppmis.values(), vect_lens))
        self.word2rowindex = dict(imap(lambda (i,k): (k,i), enumerate(ppmis.keys())))
        
    def project(self, item, sparse = False):
        """ Returns a full, none-sparse vector representation of a item
        """
        row = self.rows[self.word2rowindex[item]]
        if sparse:
            return row
        return gensim_to_numpy_array([row], self.num_terms, None)[0]
    
    """ Interface methods """
    def __iter__(self):
        for row in self.rows:
            yield row
    
    def __len__(self):
        return len(self.rows)
    """ End Interface methods """

class PpmiLatentWordVectors(ProjectionABC.ProjectionABC):
    def __init__(self, tokenized_docs, num_topics = 100):
        self.corpus = PpmiWordVectors(tokenized_docs)
        self.lsa = LsiModel(self.corpus, num_topics, self.corpus.id2word)
        distance_matrix = self.lsa[self.corpus]
        
        def gensim_to_vector(gensim_vect):
            return map(lambda (id, val): val, gensim_vect)
 
        self.rows = map(gensim_to_vector, distance_matrix)
    
    def project(self, item):
        ix = self.corpus.word2rowindex[item]
        return self.rows[ix]

if __name__ == "__main__":
    import GwData as d
    import WordTokenizer as t
    
    data = d.GwData()
    tokenized_docs = t.tokenize(data.documents, spelling_correct = False)
    
    model = PpmiLatentWordVectors(tokenized_docs)
    vector = model.project("essay")
    pass