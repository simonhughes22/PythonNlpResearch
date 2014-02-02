from nltk import PorterStemmer
import numpy as np
import ProjectionABC

class Embeddings(ProjectionABC.ProjectionABC):
    
    def __init__(self, upper = 1.0, lower = 0.0, stem = False):
        dir = "C:\Users\simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\PyDevNLPLibrary\src\DeepLearning\\"

        if stem:
            self.__stemmer__ = PorterStemmer()
            self.stem = lambda wd: self.__stemmer__.stem(wd)
        else:
            self.stem = lambda wd: wd
            
        #fEmbeddings = open(dir + "embeddings.gw.txt", "r+")
        fEmbeddings = open(dir + "embeddings.txt", "r+")
        lines = fEmbeddings.readlines()
        distance_matrix = np.distance_matrix(np.ones((len(lines), 50), 'float'))
    
        for i,line in enumerate(lines):
            split = line.split()
            if len(split) != 50:
                raise Exception("Line is not the expected length (50)")
            row = [float(s) for s in split]
            distance_matrix[i] = row
        fEmbeddings.close()
        
        sz = upper - lower
        # Normalize distance_matrix
        
        for c in range(50):
            col = distance_matrix[:,c]
            maxVal = max(col)
            minVal = min(col)
            diff = maxVal - minVal
            ncol = sz * (((col - minVal) / diff) + lower)
            distance_matrix[:,c] = np.distance_matrix(ncol)
        
        l = []
        for i in range(distance_matrix.shape[0]):
            r = distance_matrix[i,:].flatten().tolist()[0]
            l.append(r)
        
        words = dict()
        i = 0
        
        #fWords = open(dir + "words.gw.lst", "r+")
        fWords = open(dir + "words.lst", "r+")
        for wd in fWords.readlines():
            w = self.__clean__(wd)
            words[w] = i
            i += 1
        fWords.close()
        
        self.words = words
        self.distance_matrix = distance_matrix
    
    def __clean__(self, wd):
        w = wd.strip().lower()
        return self.stem(w)
    
    def project(self, word):
        wd = self.__clean__(word)
        if not wd in self.words:
            raise Exception("Word not in embeddings")
        return self.distance_matrix[self.words[wd]].flatten().tolist()[0]
    
    def project(self, item):
        if type(item) == type(""):
            return self.project(item)
        
        l = []
        for w in item:
            if w in self.words:
                l.append( self.project(w))
        return l
    
if __name__ == "__main__":
    import GwData
    import TfIdf
    import WordTokenizer

    e = Embeddings()

    d = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(d.documents, min_word_count=1, stem = False, remove_stop_words = False)
    tf = TfIdf.TfIdf(tokenized_docs)
   
    ewds = set(e.words)
    
    dwds = set([w for w in tf.id2Word.values()])    
    pass