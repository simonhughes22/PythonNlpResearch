def compute_skip_grams(tokens, max_span_len = None):
    """
        tokens          :   list[str]
                                list of tokens to compute binary skip grams over
        max_span_len    :   int
                                maximum number of tokens spanned by skip gram

        Computes binary skip grams (two words), spanning up to max_span_len words
    """

    if max_span_len == None:
        max_span_len = len(tokens)
    
    if len(tokens) < 2:
        return None
    
    skip_grams = []
    # unigrams
    for skip_gram_span in range(2, max_span_len + 1):
        for start in range(0, len(tokens) - skip_gram_span + 1):
            end = start + skip_gram_span -1
            words = [tokens[start], tokens[end]]
            skip_grams.append(words)
    return skip_grams

def skip_gram_matches(skip_gram, sentence):
    """
    skip_gram   :   list of str
                        skip gram to test
    sentence    :   list of str
                        sentence to test skip gram on
    returns     :   bool

        Returns true if the skip gram matches the sentence,
        otherwise false
    """
    is_match = False
    queue = skip_gram[:]
    for term in sentence:
        if term == queue[0]:
            if len(queue) == 1:
                is_match = True
                break
            queue = queue[1:]
    return is_match

if __name__ == "__main__":
    
    def print_ngrams(skip_grams):
        for n in sorted(skip_grams):
            print len(n), n
        
    tokens = range(1,5)
    skip_grams = compute_skip_grams(tokens)
    print "5"
    print tokens
    print_ngrams(skip_grams)
    
    print "\n2"
    skip_grams = compute_skip_grams(tokens, 2)
    print tokens
    print_ngrams(skip_grams)
    
    print "\n3"
    skip_grams = compute_skip_grams(tokens, 3)
    print tokens
    print_ngrams(skip_grams)
    
    print "\n4"
    skip_grams = compute_skip_grams(tokens, 5)
    print tokens
    print_ngrams(skip_grams)
