import nltk

class PosTagger(object):
    """Tags POS given a list of tokens"""

    def tag(self, tokenized_doc):
        """ a set of tokens """
        
        """ Note: I suspect that These tokens should not be stemmed 
            and capitalization should be preserved 
        """ 
        return nltk.pos_tag(tokenized_doc)

if __name__ == "__main__":

    tagger = PosTagger()
    print(tagger.tag("A man was walking his dog in the rain .".split()))
    print(tagger.tag("The dog chased the cat up the tall oak tree.".split()))