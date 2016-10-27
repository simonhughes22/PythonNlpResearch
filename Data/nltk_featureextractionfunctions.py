# coding=utf-8
from NgramGenerator import compute_ngrams
from Decorators import memoize
from SpaCyParserWrapper import Parser
from featureextractionfunctions import attach_function_identifier
import PosTagger

from nltk import PorterStemmer
stemmer = PorterStemmer()

@memoize
def stem(word):
    return stemmer.stem(word)

__pos_tagger__ = PosTagger.PosTagger()
@memoize
def __tag__(tokens):
    return __pos_tagger__.tag(tokens)

# initialize spaCy parser
# http://honnibal.github.io/spaCy/quickstart.html#install
parser = Parser()
@memoize
def __parse__(tokens):
    return parser.parse(tokens)

__START__ = "<START>"
__END__   = "<END>"

def fact_composite_feature_extractor(extraction_fns):
    def composite_feature_extractor(tokens, idx):
        feats = []
        for fn in extraction_fns:
            fts = fn(tokens, idx)
            feats.extend(fts)
        return feats
    return composite_feature_extractor

def fact_extract_positional_word_features(offset, positional=True, stem_words=False):

    lcls = locals()

    if stem_words:
        map_fn = stem
        prefix = "STEM_"
    else: # identity transform
        map_fn = lambda x : x
        prefix = ""

    def extract_positional_word_features(tokens, idx):
        feats = []
        start = idx - offset
        stop  = idx + offset

        end = len(tokens) - 1
        for i in range(start, stop+1):
            if positional:
                relative_offset = str(i - idx)
            else:
                relative_offset = "BOW"
            if i < 0:
                feats.append(prefix + "WD:" + relative_offset + "->" + __START__ )
            elif i > end:
                feats.append(prefix + "WD:" + relative_offset + "->" + __END__)
            else:
                offset_word = map_fn(tokens[i])
                feats.append(prefix + "WD:" + relative_offset + "->" + offset_word)
        return feats

    attach_function_identifier(extract_positional_word_features, lcls)
    return extract_positional_word_features

def fact_extract_ngram_features(offset, ngram_size, positional=True, stem_words=False):
    lcls = locals()

    def extract_ngram_features(tokens, idx):
        feats = []
        end = len(tokens) - 1

        # fix to within bounds only
        start = max(0, idx - offset)
        stop = min(end, idx + offset)

        prefix = ""
        if stem_words:
            prefix = "STEM_"

        window = list(tokens[start:stop + 1])
        if stem_words:
            window = map(lambda x: stem(x), window)

        if idx < offset:
            diff = offset - idx
            for i in range(diff):
                window.insert(0, __START__)
        if idx + offset > end:
            diff = idx + offset - end
            for i in range(diff):
                window.append(__END__)

        ngrams = compute_ngrams(window, ngram_size, ngram_size)
        str_num_ngrams = str(ngram_size)

        for i, offset_ngram in enumerate(ngrams):
            if positional:
                relative_offset = str(i - offset)
            else:
                relative_offset = "BOW"
            str_ngram = ",".join(offset_ngram)
            feats.append(prefix + "POS_" + str_num_ngrams + "_GRAMS:" + relative_offset + "->" + str_ngram)
        return feats

    attach_function_identifier(extract_ngram_features, lcls)
    return extract_ngram_features

def fact_extract_positional_POS_tags(offset, positional=True):

    lcls = locals()
    prefix = "POS-TAG_STEM_"

    def extract_positional_POS_tags(tokens, idx):
        feats = []
        start = idx - offset
        stop  = idx + offset

        end = len(tokens) - 1
        tag_pairs = __tag__(tokens)
        tags = zip(*tag_pairs)[1]

        for i in range(start, stop+1):
            if positional:
                relative_offset = str(i - idx)
            else:
                relative_offset = "BOW"
            if i < 0:
                feats.append(prefix + "WD:" + relative_offset + "->" + __START__ )
            elif i > end:
                feats.append(prefix + "WD:" + relative_offset + "->" + __END__)
            else:
                tag = tags[i]
                feats.append(prefix + "WD:" + relative_offset + "->" + tag)
        return feats

    attach_function_identifier(extract_positional_POS_tags, lcls)
    return extract_positional_POS_tags

@memoize
def __brown_clusters__(tokens):
    return parser.brown_cluster(tokens)

def extract_brown_cluster(tokens, idx):
    clusters = __brown_clusters__(tokens)
    return ["BRN_CL:" + clusters[idx]]

def extract_dependency_relation(tokens, idx):
    relations = __parse__(tokens)
    relation = relations[idx]
    bin_relations = relation.binary_relations()
    feats = []
    for bin_rel in bin_relations:
        feats.append("DEP_RELN[" + bin_rel.relation + "]:" + bin_rel.head + "->" + bin_rel.child)
    return feats

if __name__ == "__main__":

    def test(sentence, fn):
        print(sentence)
        print(fn.func_name)
        tokens = sentence.split(" ")
        for i, tok in enumerate(tokens):
            feats = fn(tokens, i)
            feats_str = ",".join(map(str, feats))
            print(tok.ljust(20) + "  :  " + feats_str)
        print("")


    test("the cat was sitting on the oven", extract_dependency_relation)
    test("the cat was sitting on the oven", extract_brown_cluster)
    test("the cat was sitting on the oven", fact_extract_positional_POS_tags(1, positional=True))
    test("the cat was sitting on the oven", fact_extract_positional_word_features(2, positional=True, stem_words=False))
    test("the cat was sitting on the oven", fact_extract_positional_word_features(1, positional=False, stem_words=True))
    test("the cat was sitting on the oven", fact_extract_ngram_features(1, 3, positional=True, stem_words=False))
