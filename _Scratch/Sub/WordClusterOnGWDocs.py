__author__ = 'simon.hughes'

import GwData
import GwLargeCorpus
import WordTokenizer
from WordClusterer import WordCluster
from LatentWordVectors import LatentWordVectors

""" GW Data """

GW_MIN_DF = 2

data = GwData.GwData()

tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count=GW_MIN_DF, stem=False, lemmatize=False)
tokenized_docs = [t for t in tokenized_docs if len(t) > 0]

projector_fn = LatentWordVectors.LsaTfIdfSpace
clusterer = WordCluster(tokenized_docs, num_topics=100, min_doc_freq=GW_MIN_DF, projector_fn=projector_fn, aggregation_method="sentence")

""" GW Large Corpus """

LARGE_MIN_DF = 20

projector_fn = LatentWordVectors.LsaTfIdfSpace

large_corpus = GwLargeCorpus.GwLargeCorpus(tokenize=False)
large_tokenized_docs = WordTokenizer.tokenize(large_corpus.documents, min_word_count=LARGE_MIN_DF, stem=False, lemmatize=False)

def remove_bad_chars(doc):
    return [t for t in doc if t.isalnum()]

def removeNonAscii(s):
    return "".join(filter(lambda x: ord(x)<128, s))

large_tokenized_docs = [[removeNonAscii(w) for w in doc] for doc in large_tokenized_docs]
large_tokenized_docs = [doc for doc in large_tokenized_docs if len(doc) > 0]

""" This works pretty well """
large_clusterer  = WordCluster(large_tokenized_docs, num_topics=100, min_doc_freq=LARGE_MIN_DF,
                               projector_fn=projector_fn, aggregation_method="sentence")

window_clusterer = WordCluster(large_tokenized_docs, num_topics=100, min_doc_freq=LARGE_MIN_DF,
                               projector_fn=projector_fn, aggregation_method="window:5")
