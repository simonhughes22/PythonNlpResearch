import GwLargeCorpus

import py_word2vec
import GwData
import WordTokenizer

#py_word2vec.word2vec('C:/data/text8', 'C:/data/text8.bin', size=300)

data = GwData.GwData()

tokenized_docs = WordTokenizer.tokenize(data.documents[:1000], min_word_count=5, stem=True, lemmatize=False)
#tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count=5, stem=True, lemmatize=False)
tokenized_docs = [t for t in tokenized_docs if len(t) > 0]

w = py_word2vec.Word2Vec(tokenized_docs,  size=100, window=5, min_count=5, workers=1)

""" Large Corpus """

LARGE_MIN_DF = 20
large_corpus = GwLargeCorpus.GwLargeCorpus(tokenize=False)
large_tokenized_docs = WordTokenizer.tokenize(large_corpus.documents, min_word_count=LARGE_MIN_DF, stem=False, lemmatize=False)

def remove_bad_chars(doc):
    return [t for t in doc if t.isalnum()]

def removeNonAscii(s):
    return "".join(filter(lambda x: ord(x)<128, s))

large_tokenized_docs = [[removeNonAscii(w) for w in doc] for doc in large_tokenized_docs]
large_tokenized_docs = [doc for doc in large_tokenized_docs if len(doc) > 0]

w_large = py_word2vec.Word2Vec(large_tokenized_docs, size=100, window=5, min_count=5)

""" Usage """
#w_large.most_similar(positive=["co2"], negative=[], topn=10)