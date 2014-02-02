from nltk import PorterStemmer
import numpy as np

class Embeddings(object):
    
    def __init__(self, upper = 1.0, lower = 0.0, stem = False, suppress_errors = False):
        dir = "C:\Users\simon\Dropbox\PhD\Code\NLP Library\NlpLibrary\PyDevNLPLibrary\PyDevNLPLibrary\src\DeepLearning\\"

        self.suppress_errors = suppress_errors
        if stem:
            self.__stemmer__ = PorterStemmer()
            self.stem = lambda wd: self.__stemmer__.stem(wd)
        else:
            self.stem = lambda wd: wd
            
        #fEmbeddings = open(dir + "embeddings.gw.txt", "r+")
        fEmbeddings = open(dir + "embeddings.gw.txt", "r+")
        lines = fEmbeddings.readlines()
        matrix = np.matrix(np.ones((len(lines), 50), 'float'))
    
        for i,line in enumerate(lines):
            split = line.split()
            if len(split) != 50:
                raise Exception("Line is not the expected length (50)")
            row = [float(s) for s in split]
            matrix[i] = row
        fEmbeddings.close()
        
        sz = upper - lower
        # Normalize matrix
        
        for c in range(50):
            col = matrix[:,c]
            maxVal = max(col)
            minVal = min(col)
            diff = maxVal - minVal
            ncol = sz * (((col - minVal) / diff) + lower)
            matrix[:,c] = np.matrix(ncol)
        
        lmatrix = []
        for i in range(matrix.shape[0]):
            r = matrix[i,:].flatten().tolist()[0]
            lmatrix.append(r)
        
        words = dict()
        i = 0
        
        #fWords = open(dir + "words.gw.lst", "r+")
        fWords = open(dir + "words.gw.lst", "r+")
        for wd in fWords.readlines():
            w = self.__clean__(wd)
            words[w] = i
            i += 1
        fWords.close()
        
        self.words = words
        self.matrix = lmatrix
    
    def __clean__(self, wd):
        w = wd.strip().lower()
        #does a no-op if not set to stem
        return self.stem(w)
    
    def __project__(self, word):
        wd = self.__clean__(word)
        if wd not in self.words:
            if not self.suppress_errors:
                raise Exception("Word not in embeddings: " + word)
            else:
                return []
        return self.matrix[self.words[wd]]
    
    def project(self, item):
        if type(item) == type(""):
            return self.__project__(item)
        
        l = []
        for w in item:
            if w in self.words:
                l.append( self.__project__(w))
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