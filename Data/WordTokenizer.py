from SpellingCorrector import *
from collections import defaultdict
from nltk import PorterStemmer, tokenize as tkn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import sys

class WordTokenizer(object):
    
    def __init__(self, stem = True, lemmatize = False, remove_stop_words = True, min_word_count = 2, spelling_correct = True, number_fn = None):
        
        self.stem = stem
        self.lemmatize = lemmatize
        
        if stem and lemmatize:
            raise Exception("Cannot stem and lemmatize")
        
        self.min_word_count = min_word_count
        
        if self.stem:
            self.__stemmer__ = PorterStemmer()
        
        if self.lemmatize:
            self.__lemmatizer__ = WordNetLemmatizer()
        
        self.spelling_correct = spelling_correct
        self.__stop_words__ = set()
        if remove_stop_words:
            self.__stop_words__ = set(stopwords.words())

        self.__special_tokens__ = {
                    'quote-mark': "'", 
                    'dash':  "-", 
                    'period' : ".", 
                    'lparen' : "(", 
                    'rparen' : ")",
                    'semicolon' : ";", 
                    'colon': ":" , 
                    'comma' : ",", 
                    'slash': "/" , 
                    'exclpoint' : "!", 
                    'question-mark' : "?"
        }
        self.punctuation = set(self.__special_tokens__.values())
        self.to_replace =  [":", "|", '#', '\x85', '@', '&', '\xb0', '\xba', '\xa0', "\\", "/", "-"]
        self.to_pad = [".", ",", "!", "?", "="]
        
        if number_fn == None:
            number_fn = lambda i:i
        
        self.number_fn = number_fn

    def __extract_words__(self, sentence):
        cleanedSentence = sentence.lower()
        
        for bad_str in self.to_replace:
            if cleanedSentence.find(bad_str) > -1:
                cleanedSentence = cleanedSentence.replace(bad_str, " ")
        
        for pad_str in self.to_pad:
            if cleanedSentence.find(pad_str) > -1:
                cleanedSentence = cleanedSentence.replace(pad_str, " " + pad_str + " ")

        return [self.number_fn(w) for w in tkn.word_tokenize(cleanedSentence.strip()) if len(w) > 0]
    
    def __process_word__(self, word, tokens, unique_tokens, docFreq):

        if word.endswith("n't"):
            self.__process_word__(word[:-3], tokens, unique_tokens, docFreq)
            self.__process_word__("not", tokens, unique_tokens, docFreq)
            return

        token = word
        if word in self.__special_tokens__:
            token = self.__special_tokens__[word]            
        
        if token in self.punctuation:
            return
                
        if token in self.__stop_words__:
            return
        
        tokens.append(token)
        if not token in unique_tokens:
            unique_tokens.add(token)
            docFreq[token] = docFreq[token]  + 1
    
    def __process_words__(self, words, docFreq):
        unique_tokens = set()
        tokens = []
        for wd in words:
            self.__process_word__(wd, tokens, unique_tokens, docFreq)
        return tokens

    def __tally_document_frequency__(self, tokens, docFreq):
        unique_tokens = set(tokens)
        for dbnetwork in unique_tokens:
            docFreq[dbnetwork] = docFreq[dbnetwork] + 1
        return docFreq

    def __remove_words_with_freq_less_than__(self, lTokens, docFreq):
        infrequentTokens = set([wd for wd, freq in docFreq.items()
                               if freq < self.min_word_count])
        return [[dbnetwork 
                 for dbnetwork in s
                 if not dbnetwork in infrequentTokens]
                for s in lTokens]
    
    # Public Methods
    def tokenize(self, sentences):
        """ Takes a list of sentences strings
            and tokenizes them   
        """

        #logging.log(logging.INFO, "Tokenizing %i sentences", len(sentences))

        # Frequency of words in each sentence
        docFreq = defaultdict(int)
        
        lTokens = []
        for sentence in sentences:
            wds = self.__extract_words__(sentence)
            tokens = self.__process_words__(wds, docFreq)
            #docFreq = self.__tally_document_frequency__(tokens, docFreq)
               
            # Tally word doc freq
            lTokens.append(tokens)
        
        if self.spelling_correct:
            sc = SpellingCorrector()
            def correct(word):
                # expect miss-spellings to occur infrequently
                if docFreq[word] >= self.min_word_count:
                    return word
                return sc.correct(word)
            lTokens, docFreq = self.__process_tokens__(lTokens, correct)    
        
        if self.stem:
            def stem_word(word):
                return self.__stemmer__.stem(word)
            lTokens, docFreq = self.__process_tokens__(lTokens, stem_word)
        
        if self.lemmatize:
            def lemmatize_word(word):
                return self.__lemmatizer__.lemmatize(word)
            lTokens, docFreq = self.__process_tokens__(lTokens, lemmatize_word)

        if self.min_word_count > 0:
            return self.__remove_words_with_freq_less_than__(lTokens, docFreq)
        else:
            return lTokens

    def __process_tokens__(self, lTokens, fn):
        lst_tokens = []
        docFreq = defaultdict(int)
        
        for tokens in lTokens:
            doc = []
            unique_tokens = set()
            for token in tokens:
                processed = fn(token)
                if len(processed) == 0:
                    continue
                if processed in self.__stop_words__:
                    continue
                if not processed in unique_tokens:
                    docFreq[processed] += 1
                doc.append(processed)
            lst_tokens.append(doc)
        
        return (lst_tokens, docFreq)

    def toFile(self, tokenized_documents, fileName):
        try:
            handle = open(fileName, mode = "w+")
            for sentence in tokenized_documents:
                line = " ".join(sentence)
                handle.write(line)
                handle.write("\n")
            handle.close()
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            print "Unexpected error:", sys.exc_info()[0]

def tokenize(documents, min_word_count = 5, stem = True, lemmatize = False, remove_stop_words = True, spelling_correct = True, number_fn = None):
    tokenizer = WordTokenizer(min_word_count = min_word_count, stem = stem, lemmatize = lemmatize, remove_stop_words = remove_stop_words, spelling_correct= spelling_correct, number_fn = number_fn)
    return tokenizer.tokenize(documents)


if __name__ == "__main__":
    import GwData
    data = GwData.GwData()
    
    tokens = tokenize(data.documents, stem = True, remove_stop_words = True, spelling_correct = True)
    pass
    