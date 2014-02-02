import nltk

class PosTagger(object):
    """Tags POS given a list of tokens"""

    def tag(self, tokenized_docs):
        """ Tag a set of tokens """
        
        """ Note: I suspect that These tokens should not be stemmed 
            and capitalization should be preserved 
        """ 
        return [nltk.pos_tag(doc) for doc in tokenized_docs]
