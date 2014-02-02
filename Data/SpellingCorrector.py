import re, collections
import os
import Settings

class SpellingCorrector(object):

    """ 
        This is a simple spelling corrector taking from Peter Norvig's site
    """    
    def __init__(self):

        settings = Settings.Settings()
        large_text_file = settings.data_directory + "big.txt"
        dictionary_file = settings.data_directory + "words.lst"

        self.nwords = self.train(self.words(file(large_text_file).read()))

        with open(dictionary_file, "r+") as f:
            words_in_dict = f.readlines()

        for line in words_in_dict:
            self.nwords[line.lower().strip()] += 1
        
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    def words(self, text): 
        return re.findall('[a-z]+', text.lower()) 

    def train(self,features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)
    
    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.nwords)
    
    def known(self, words): return set(w for w in words if w in self.nwords)
    
    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
        return max(candidates, key=self.nwords.get)
    
if __name__ == "__main__":
    sc = SpellingCorrector()
    
    print "appe", sc.correct("appe")
    print "apple", sc.correct("apple")
    print "aple", sc.correct("aple")
    print "appl", sc.correct("appl")
    print "appple", sc.correct("appple")
    