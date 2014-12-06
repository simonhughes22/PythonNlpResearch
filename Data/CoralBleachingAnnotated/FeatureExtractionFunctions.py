from NgramGenerator import compute_ngrams

""" POSITIONAL SINGLE WORDS
"""

def fact_extract_positional_word_features(offset):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn(input, val=1):
        return extract_positional_word_features(offset, input, val)
    return fn

def extract_positional_word_features(offset, input, val = 1):
    """ offset      :   int
                           the number of words either side of the input to extract features from
        input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    feats = {}
    start = input.wordix - offset
    stop  = input.wordix + offset

    end = len(input.sentence) - 1
    for i in range(start, stop+1):
        offset_word = input.sentence[i]
        relative_offset = str(input.wordix - start)
        if i < 0:
            feats["SENT_START:" + relative_offset] = val
        elif i > end:
            feats["SENT_END:" + relative_offset] = val
        else:
            feats["WD:" + relative_offset + "->" + offset_word] = val
    return feats

""" POSITIONAL NGRAMS
"""
def fact_extract_ngram_features(offset, ngram_size):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        ngram_size  :   int
                            the size of the ngrams
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset and ngram size
    def fn(input, val=1):
        return extract_ngram_features(offset, ngram_size, input, val)
    return fn

def extract_ngram_features(offset, ngram_size, input, val = 1):
    """ offset      :   int
                           the number of words either side of the input to extract features from
        ngram_size  :   int
                            the size of the ngrams
        input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    feats = {}
    end = len(input.sentence) - 1 - (ngram_size -1) # last ngram

    # fix to within bounds only
    start = max(0, input.wordix - offset)
    stop  = min(end, input.wordix + offset)

    ngrams = compute_ngrams(input.sentence, ngram_size, ngram_size)
    str_num_ngrams = str(ngram_size)

    for i in range(start, stop+1):
        offset_ngram = ngrams[i]
        relative_offset = str(start - input.wordix)
        str_ngram = ",".join(offset_ngram)
        feats["POS_" + str_num_ngrams + "GRAMS:" + relative_offset + "->" + str_ngram] = val
    return feats


