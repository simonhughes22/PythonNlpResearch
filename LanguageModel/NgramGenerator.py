from typing import List

def compute_ngrams(tokens: List[str], max_len: int = None, min_len: int = 1)-> List[List[str]]:
    
    if max_len == None:
        max_len = len(tokens)
    
    ngrams = []
    # unigrams
    for ngram_size in range(min_len, max_len + 1):
        for start in range(0, len(tokens) - ngram_size + 1):
            end = start + ngram_size -1
            words = []
            for i in range(start, end + 1):
                words.append(tokens[i])
            ngrams.append(words)
    return ngrams

if __name__ == "__main__":
    
    def print_ngrams(ngrams):
        for n in ngrams:
            print(len(n), n)
        
    tokens = range(1,5)
    ngrams = compute_ngrams(tokens)
    print_ngrams(ngrams)
    
    print("")
    ngrams = compute_ngrams(tokens, 3)
    print_ngrams(ngrams)

    print("")
    ngrams = compute_ngrams(tokens, 3, 3)
    print_ngrams(ngrams)

    print("")
    ngrams = compute_ngrams(tokens, 4, 2)
    print_ngrams(ngrams)

