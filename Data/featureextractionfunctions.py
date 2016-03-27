from NgramGenerator import compute_ngrams
from Decorators import memoize
from SpaCyParserWrapper import Parser
import PosTagger

from nltk import PorterStemmer
stemmer = PorterStemmer()

@memoize
def stem(word):
    return stemmer.stem(word)

# initialize spaCy parser
# http://honnibal.github.io/spaCy/quickstart.html#install
parser = Parser()

"""
TODO - SpaCy - Try brown cluster labels
               Try pos tags tok.pos_
               Try the 2nd tag type tok.tag_
               Try the dep parse tag
               Try the full dep parse relation as features e.g. p3. https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf
               Try the dep word embedding

     >>>>>>> DO for target word ONLY, and for positional features !!!!
"""

def extract_dependency_children(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["CHILD_DEP:" + bin_rel.relation + "->" + bin_rel.child] = val
    return feats

def extract_dependency_child_words(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["CHILD_WORD_DEP:" + bin_rel.child] = val
    return feats

def extract_dependency_children_plus_target(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["CHILD_DEP_TGT[" + input.sentence[input.wordix] + "]:" + bin_rel.relation + bin_rel.child] = val
    return feats

def extract_dependency_head(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    feats = { "HEAD_DEP:" + relation.head + "->" + relation.relation : val }
    return feats

def extract_dependency_head_word(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    feats = { "HEAD_WORD_DEP:" + relation.head: val }
    return feats

def extract_dependency_head_plus_target(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    feats = { "HEAD_DEP_TGT[" + input.sentence[input.wordix] + "]:" + relation.head + "->" + relation.relation: val }
    return feats

def extract_dependency_relation(input, val=1):
    relations = parser.parse(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["DEP_RELN[" + bin_rel.relation + "]:" + bin_rel.head + "->" + bin_rel.child] = val
    return feats

""" POSITIONAL SINGLE WORDS
"""

__START__ = "<START>"
__END__   = "<END>"


def fact_extract_positional_head_word_features(offset):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_pos_hd_wd_feats(input, val=1):
        return extract_positional_head_word_features(offset, input, val)
    return fn_pos_hd_wd_feats

def extract_positional_head_word_features(offset, input, val = 1):
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

    relations = parser.parse(input.sentence)

    end = len(input.sentence) - 1
    for i in range(start, stop+1):
        relative_offset = str(i - input.wordix)
        if i < 0:
            feats["WD_HEAD" + __START__ + ":" + relative_offset] = val
        elif i > end:
            feats["WD_HEAD" + __END__ + ":" + relative_offset] = val
        else:
            offset_relation = relations[i]
            feats["WD_HEAD:" + relative_offset + "->" + offset_relation.head] = val
    return feats

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
            feats["WD" +__START__ + ":" + relative_offset] = val
        elif i > end:
            feats["WD" +__END__ + ":" + relative_offset] = val
        else:
            offset_word = input.sentence[i]
            feats["WD:" + relative_offset + "->" + offset_word] = val
    return feats

def fact_extract_positional_word_features_stemmed(offset):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_pos_wd_feats_stemmed(input, val=1):
        return extract_positional_word_features_stemmed(offset, input, val)
    return fn_pos_wd_feats_stemmed # recently renamed for mongodob logging

def extract_positional_word_features_stemmed(offset, input, val = 1):
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
            feats["WD" +__START__ + ":" + relative_offset] = val
        elif i > end:
            feats["WD" +__END__ + ":" + relative_offset] = val
        else:
            offset_word = stem(input.sentence[i])
            feats["WD:" + relative_offset + "->" + offset_word] = val
    return feats

def fact_extract_first_3_chars(offset):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_extract_first_3_chars(input, val=1):
        return extract_first_3_chars(offset, input, val)
    return fn_extract_first_3_chars # recently renamed for mongodob logging

def extract_first_3_chars(offset, input, val = 1):
    """ offset      :   int
                           the number of words either side of the input to extract features from
        input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    feats = {}
    start = input.wordix - offset
    stop = input.wordix + offset

    end = len(input.sentence) - 1
    for i in range(start, stop + 1):
        relative_offset = str(i - input.wordix)
        #only bother when there is a word within window
        if i >= 0 and i < end:
            offset_word = input.sentence[i].strip()
            feats["First3:" + relative_offset + "->" + offset_word[:3]] = val
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

def fact_extract_ngram_features_stemmed(offset, ngram_size):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        ngram_size  :   int
                            the size of the ngrams
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset and ngram size
    def fn_ngram_feat_stemmed(input, val=1):
        return extract_ngram_features_stemmed(offset, ngram_size, input, val)
    return fn_ngram_feat_stemmed

def extract_ngram_features_stemmed(offset, ngram_size, input, val = 1):
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
    window = map(stem, window)
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
    def fn_pos_POS_feats_stemmed(input, val=1):
        return extract_positional_POS_features(offset, input, val)
    return fn_pos_POS_feats_stemmed

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
            feats["POS_TAG" + __START__ + ":" + relative_offset] = val
        elif i > end:
            feats["POS_TAG" + __END__ + ":" + relative_offset] = val
        else:
            offset_word = tags[i]
            feats["POS_TAG:" + relative_offset + "->" + offset_word] = val
    return feats

def extract_POS_TAG(input, val = 1):
    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    tag_pairs = __tag__(input.sentence)
    tags = zip(*tag_pairs)[1]
    feats = {"POS_TAG_ONLY:" + tags[input.wordix] : val }
    return feats

def extract_POS_TAG_PLUS_WORD(input, val = 1):
    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    tag_pairs = __tag__(input.sentence)
    tags = zip(*tag_pairs)[1]
    feats = {"POS_TAG_WD:" + tags[input.wordix] + "-" + input.sentence[input.wordix] : val }
    return feats

def fact_extract_positional_POS_features_plus_word(offset):
    """ offset      :   int
                            the number of words either side of the input to extract POS features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_pos_POS_feats_stemmed_plus_word(input, val=1):
        return extract_positional_POS_features_plus_word(offset, input, val)
    return fn_pos_POS_feats_stemmed_plus_word

def extract_positional_POS_features_plus_word(offset, input, val = 1):
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
            feats["POS_TAG_Posn_Word" + __START__ + ":" + relative_offset] = val
        elif i > end:
            feats["POS_TAG_Posn_Word" + __END__ + ":" + relative_offset] = val
        else:
            offset_tag = tags[i]
            feats["POS_TAG_Posn_Word:" + relative_offset + "->" + offset_tag + ":" + input.sentence[i]] = val
    return feats

def extract_brown_cluster(input, val = 1):
    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    clusters = parser.brown_cluster(input.sentence)
    feats = {"BRN_CL:" + clusters[input.wordix] : val }
    return feats

def extract_brown_cluster_plus_word(input, val = 1):
    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    clusters = parser.brown_cluster(input.sentence)
    feats = {"BRN_CL:" + clusters[input.wordix] + "-" + input.sentence[input.wordix] : val }
    return feats

def extract_POS_TAG_PLUS_WORD(input, val = 1):
    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    tag_pairs = __tag__(input.sentence)
    tags = zip(*tag_pairs)[1]
    feats = {"POS_TAG_WD:" + tags[input.wordix] + "-" + input.sentence[input.wordix] : val }
    return feats

def __vector_to_dict_(vector, prefix):
    """

    @param vector:  1D numpy array
    @param prefix:  str
    @return:        dict
    """
    d = dict()
    for i, v in enumerate(vector):
        if v > 0.0:
            d[prefix + ":" + str(i)] = v
    return d

def extract_single_dependency_vector(input, val = 1):
    vectors = parser.dep_vector(input.sentence)
    return __vector_to_dict_(vectors[input.wordix], "1_VEC")

def fact_extract_positional_dependency_vectors(offset):
    """ offset      :   int
                            the number of words either side of the input to extract POS features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    def fn_extract_positional_dependency_vectors(input, val=1):
        return extract_extract_positional_dependency_vectors(offset, input, val)
    return fn_extract_positional_dependency_vectors

def extract_extract_positional_dependency_vectors(offset, input, val = 1):
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
    vectors = parser.dep_vector(input.sentence)

    PREFIX = "POS_DEP_VEC"
    for i in range(start, stop+1):
        relative_offset = str(i - input.wordix)
        if i < 0:
            feats[PREFIX + __START__ + ":" + relative_offset] = 1.0
        elif i > end:
            feats[PREFIX + __END__ +   ":" + relative_offset] = 1.0
        else:
            positional_prefix = PREFIX + ":" + relative_offset
            d = __vector_to_dict_(vectors[input.wordix], positional_prefix)
            feats.update(d)
    return feats