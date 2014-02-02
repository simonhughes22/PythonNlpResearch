from collections import defaultdict
from functools import partial
from itertools import combinations, imap, chain
from numpy.oldnumeric.random_array import permutation

def compute_id2_word(tokenized_docs):
    
    id = 0
    id2word = dict()
    word2id = dict()
    
    for doc in tokenized_docs:
        for word in doc:
            if word not in word2id:
                word2id[word] = id
                id2word[id] = word
                id +=1
    return (id2word, word2id)

def word_tally(tokenized_docs):
    tally = defaultdict(int)
    
    def tally_word(word):
        tally[word] += 1
        
    map(lambda tokens: tally_word(tokens), tokenized_docs)
    return tally    

def cooccurence_counts(tokenized_docs, initial_count = 0):
    """ 
        Compute co-occurence counts of terms in the tokenized_docs, using
        initial count to smooth counts. Returns a function
        that returns a count of the word pair.
    """
    tally_by_word = defaultdict(lambda : defaultdict(lambda : initial_count))
    
    def apply_tally(pair):
        a,b = pair
        tally_by_word[a][b] += 1
        tally_by_word[b][a] += 1
    
    pairs = partial(combinations, r = 2)
    word_pairs = reduce(chain, imap(pairs, tokenized_docs), [])
    map(apply_tally, word_pairs)
    
    return tally_by_word
    
if __name__ == "__main__":
    
    tokens = [map(str, [1,2,3,4,5,6]), map(str,[1,2,3,3])]
    counts = cooccurence_counts(tokens, 0)
    
    for word in tokens[0]:
        print word, counts[word]
    