from NgramGenerator import compute_ngrams
from Decorators import memoize
from spacy.en import English
import PosTagger

# initialize spaCy parser
# http://honnibal.github.io/spaCy/quickstart.html#install
nlp = English()

@memoize
def spacy_parse(text):
    return list(nlp(unicode(text)))

"""
TODO - SpaCy - Try brown cluster labels
               Try pos tags tok.pos_
               Try the 2nd tag type tok.tag_
               Try the dep parse tag
               Try the full dep parse relation as features e.g. p3. https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf
               Try the dep word embedding

     >>>>>>> DO for target word ONLY, and for positional features !!!!
"""



""" POSITIONAL SINGLE WORDS
"""

__START__ = "<START>"
__END__   = "<END>"

def fact_extract_positional_word_features(offset):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_pos_wd_feats(input, val=1):
        return extract_positional_word_features(offset, input, val)
    return fn_pos_wd_feats

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
        relative_offset = str(i - input.wordix)
        if i < 0:
            feats[__START__ + ":" + relative_offset] = val
        elif i > end:
            feats[__END__ + ":" + relative_offset] = val
        else:
            offset_word = input.sentence[i]
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
    def fn_ngram_feat(input, val=1):
        return extract_ngram_features(offset, ngram_size, input, val)
    return fn_ngram_feat

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
    end = len(input.sentence) - 1

    # fix to within bounds only
    start = max(0, input.wordix - offset)
    stop  = min(end, input.wordix + offset)

    window = list(input.sentence[start:stop+1])
    if input.wordix < offset:
        diff = offset - input.wordix
        for i in range(diff):
            window.insert(0,__START__)
    if input.wordix + offset > end:
        diff = input.wordix + offset - end
        for i in range(diff):
            window.append(__END__)

    ngrams = compute_ngrams(window, ngram_size, ngram_size)
    str_num_ngrams = str(ngram_size)

    for i, offset_ngram in enumerate(ngrams):
        relative_offset = str(i - offset)
        str_ngram = ",".join(offset_ngram)
        feats["POS_" + str_num_ngrams + "GRAMS:" + relative_offset + "->" + str_ngram] = val

    return feats

__pos_tagger__ = PosTagger.PosTagger()
@memoize
def __tag__(tokens):
    return __pos_tagger__.tag(tokens)

def fact_extract_positional_POS_features(offset):
    """ offset      :   int
                            the number of words either side of the input to extract POS features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_pos_POS_feats(input, val=1):
        return extract_positional_POS_features(offset, input, val)
    return fn_pos_POS_feats

def extract_positional_POS_features(offset, input, val = 1):
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
    tag_pairs = __tag__(input.sentence)
    tags = zip(*tag_pairs)[1]

    for i in range(start, stop+1):
        relative_offset = str(i - input.wordix)
        if i < 0:
            feats[__START__ + ":" + relative_offset] = val
        elif i > end:
            feats[__END__ + ":" + relative_offset] = val
        else:
            offset_word = tags[i]
            feats["POS_TAG:" + relative_offset + "->" + offset_word] = val
    return feats
