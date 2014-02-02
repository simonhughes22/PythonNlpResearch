from itertools import imap
import VectorSpaceUtils as utils
import math

def Ppmi(tokenized_docs):
    
    """ Use initial count of 2 to smooth """
    cooccurence_counts = utils.cooccurence_counts(tokenized_docs, 2)
    words = cooccurence_counts.keys()
    totals = map(lambda (k,dct): (k, sum(dct.values())), cooccurence_counts.items())
    total = sum([v for k,v in totals]) * 1.0
    priors = dict(map(lambda (k,cnt): (k, v / total), totals))
    
    def ppmi_pair(a,b):
        pAB = cooccurence_counts[a][b] / total
        pA = priors[a]
        pB = priors[b]
        return max(0, math.log(pAB / (pA * pB)))
    
    def ppmi(word):
        join_cnts = cooccurence_counts[word]
        return (word, map( lambda k: (k, ppmi_pair(word, k)), join_cnts.keys()))
    
    return dict(imap(ppmi, words))
            
if __name__ == "__main__":
   
    tokens = [map(str, [1,2,3,4,5,6]), map(str,[1,2,3,3])]
    rslt = Ppmi(tokens)
    for k,v in rslt.items():
        print k, v
    