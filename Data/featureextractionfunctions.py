from NgramGenerator import compute_ngrams
from Decorators import memoize
from SpaCyParserWrapper import Parser
import PosTagger
import sys

PYTHON_VERSION = sys.version_info[0]
IS_PYTHON_3 = (PYTHON_VERSION >= 3)

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

def attach_function_identifier(fn, d):
    try:
        s = fn.func_name + "["
    except:
        s = fn.__name__ + "["
        fn.func_name = fn.__name__

    for k, v in sorted(d.items(), key = lambda tpl: tpl[0]):
        if k == fn.func_name:
            continue
        if type(v) == dict:
            continue
        s +=   "%s:%s " % (str(k), str(v))
    fn.func_name = s.strip() + "]"
    return fn

@memoize
def __parse__(tokens):
    return parser.parse(tokens)

def extract_dependency_children(input, val=1):
    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["CHILD_DEP:" + bin_rel.relation + "->" + bin_rel.child] = val
    return feats

def extract_dependency_child_words(input, val=1):
    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["CHILD_WORD_DEP:" + bin_rel.child] = val
    return feats

def extract_dependency_children_plus_target(input, val=1):
    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["CHILD_DEP_TGT[" + input.sentence[input.wordix] + "]:" + bin_rel.relation + bin_rel.child] = val
    return feats

def extract_dependency_head(input, val=1):
    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    feats = { "HEAD_DEP:" + relation.head + "->" + relation.relation : val }
    return feats

def extract_dependency_head_word(input, val=1):
    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    feats = { "HEAD_WORD_DEP:" + relation.head: val }
    return feats

def extract_dependency_head_plus_target(input, val=1):
    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    feats = { "HEAD_DEP_TGT[" + input.sentence[input.wordix] + "]:" + relation.head + "->" + relation.relation: val }
    return feats

def extract_dependency_relation(input, val=1):
    if IS_PYTHON_3:
        raise Exception("Does not work in python 3.x")

    relations = __parse__(input.sentence)
    relation = relations[input.wordix]
    bin_relations = relation.binary_relations()
    feats = {}
    for bin_rel in bin_relations:
        feats["DEP_RELN[" + bin_rel.relation + "]:" + bin_rel.head + "->" + bin_rel.child] = val
    return feats

def fact_extract_dependency_relation():
    lcls = locals()
    # curry offset
    def extract_dependency_relation_internal(input):
        return extract_dependency_relation(input=input, val=1)

    return attach_function_identifier(extract_dependency_relation_internal, lcls)

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
    lcls = locals()
    # curry offset
    def fn_pos_hd_wd_feats(input, val=1):
        return extract_positional_head_word_features(offset, input, val)
    return attach_function_identifier(fn_pos_hd_wd_feats, lcls)

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
    lcls = locals()
    # curry offset
    def fn_pos_wd_feats(input, val=1):
        return extract_positional_word_features(offset, input, val)
    return attach_function_identifier(fn_pos_wd_feats, lcls)

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
    lcls = locals()
    # curry offset
    def fn_pos_wd_feats_stemmed(input, val=1):
        return extract_positional_word_features_stemmed(offset, input, val)
    return attach_function_identifier(fn_pos_wd_feats_stemmed, lcls) # recently renamed for mongodob logging

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
    lcls = locals()
    def fn_extract_first_3_chars(input, val=1):
        return extract_first_3_chars(offset, input, val)
    return attach_function_identifier(fn_extract_first_3_chars, lcls) # recently renamed for mongodob logging

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

""" BOW NGRAMS
"""
def fact_extract_bow_ngram_features(offset, ngram_size):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        ngram_size  :   int
                            the size of the ngrams
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset and ngram size
    lcls = locals()
    def fn_bow_ngram_feat(input, val=1):
        return extract_bow_ngram_features(offset, ngram_size, input, val)
    return attach_function_identifier(fn_bow_ngram_feat, lcls)

def extract_bow_ngram_features(offset, ngram_size, input, val = 1):
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
        str_ngram = ",".join(offset_ngram)
        feats["POS_" + str_num_ngrams + "GRAMS:BOW" + "->" + str_ngram] = val

    return feats

""" POSITIONAL NGRAMS
"""
def fact_extract_positional_ngram_features(offset, ngram_size):
    """ offset      :   int
                            the number of words either side of the input to extract features from
        ngram_size  :   int
                            the size of the ngrams
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset and ngram size
    lcls = locals()
    def fn_pos_ngram_feat(input, val=1):
        return extract_positiomal_ngram_features(offset, ngram_size, input, val)

    attach_function_identifier(fn_pos_ngram_feat, lcls)
    return fn_pos_ngram_feat

def extract_positiomal_ngram_features(offset, ngram_size, input, val = 1):
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
    lcls = locals()
    # curry offset and ngram size
    def fn_pos_ngram_feat_stemmed(input, val=1):
        return extract_ngram_features_stemmed(offset, ngram_size, input, val)
    return attach_function_identifier(fn_pos_ngram_feat_stemmed, lcls)

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
    window = list(map(stem, window))
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

def fact_extract_bow_POS_features(offset):
    """ offset      :   int
                            the number of words either side of the input to extract POS features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    # curry offset
    lcls = locals()
    def fn_bow_POS_feats(input, val=1):
        return extract_bow_POS_features(offset, input, val)
    return attach_function_identifier(fn_bow_POS_feats, lcls)

def extract_bow_POS_features(offset, input, val = 1):
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
    tags = list(zip(*tag_pairs))[1]

    for i in range(start, stop+1):
        if i < 0:
            feats["POS_TAG" + __START__ + ":BOW"] = val
        elif i > end:
            feats["POS_TAG" + __END__ + ":BOW"] = val
        else:
            offset_word = tags[i]
            feats["POS_TAG:BOW" + "->" + offset_word] = val
    return feats

def fact_extract_positional_POS_features(offset):
    """ offset      :   int
                            the number of words either side of the input to extract POS features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    lcls = locals()
    # curry offset
    def fn_pos_POS_feats(input, val=1):
        return extract_positional_POS_features(offset, input, val)
    return attach_function_identifier(fn_pos_POS_feats, lcls)

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
    tags = list(zip(*tag_pairs))[1]

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
    tags = list(zip(*tag_pairs))[1]
    feats = {"POS_TAG_ONLY:" + tags[input.wordix] : val }
    return feats

def extract_POS_TAG_PLUS_WORD(input, val = 1):
    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    tag_pairs = __tag__(input.sentence)
    tags = list(zip(*tag_pairs))[1]
    feats = {"POS_TAG_WD:" + tags[input.wordix] + "-" + input.sentence[input.wordix] : val }
    return feats

def fact_extract_positional_POS_features_plus_word(offset):
    """ offset      :   int
                            the number of words either side of the input to extract POS features from
        returns     :   fn
                            feature extractor function: FeatureExtactorInput -> dict
    """
    lcls = locals()
    # curry offset
    def fn_pos_POS_feats_stemmed_plus_word(input, val=1):
        return extract_positional_POS_features_plus_word(offset, input, val)
    return attach_function_identifier(fn_pos_POS_feats_stemmed_plus_word, lcls)

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
    tags = list(zip(*tag_pairs))[1]

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

def extract_brown_cluster(input, val=1):
    if IS_PYTHON_3:
        raise Exception("Does not work in python 3.x")

    """ input      :    FeatureExtactorInput
                            input to feature extractor
        returns     :   dict
                            dictionary of features
    """

    clusters = parser.brown_cluster(input.sentence)
    feats = {"BRN_CL:" + clusters[input.wordix] : val }
    return feats

def fact_extract_brown_cluster():
    lcls = locals()
    def extract_brown_cluster_internal(input):
        return extract_brown_cluster(input=input, val=1)

    return attach_function_identifier(extract_brown_cluster_internal, lcls)

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
    tags = list(zip(*tag_pairs))[1]
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
    lcls = locals()
    # curry offset
    def fn_extract_positional_dependency_vectors(input, val=1):
        return extract_positional_dependency_vectors(offset, input, val)
    return attach_function_identifier(fn_extract_positional_dependency_vectors, lcls)

def extract_positional_dependency_vectors(offset, input, val = 1):
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

if __name__ == "__main__":

    bigram_win = fact_extract_positional_ngram_features(5, 2)
    print(bigram_win.func_name)

    from featureextractortransformer import FeatureExtractorInput

    sentence = "Mary had a little lamb".split(" ")
    tagged = [(wd, set()) for wd in sentence]
    input = FeatureExtractorInput(1, tagged,1, None)
    rels = extract_dependency_relation(input)
    for rel in rels:
        print(rel)
