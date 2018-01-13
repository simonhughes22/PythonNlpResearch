from LatentWordVectors import LatentWordVectors
import GwData
import MatrixHelper
import WordTokenizer
import logging
from sklearn import cluster, metrics
from nltk.cluster import kmeans
from collections import defaultdict
import NumberStrategy
from DictionaryHelper import *
"""
    %run "C:\Users\simon.hughes\Dropbox\PhD\Code\NLP Library\PyDevNLPLibrary\src\Experiments\GlobalWarming\WordClustering\WordClustering"
"""
def extract_clusters_from_kmeans(km, items):
    km_clusters = defaultdict(list)
    for i,lbl in enumerate(km.labels_):
        km_clusters[lbl].append(items[i])
    return map(sorted, km_clusters.values())

def docs_to_clusters(tokenized_docs, word2cluster):
    cluster_labels = []
    for doc in tokenized_docs:
        clusters = set(map(lambda wd: word2cluster[wd], doc))
        cluster_labels.append(clusters)
    return cluster_labels
    

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data = GwData.GwData()
docs = data.documents

tokenized_docs = WordTokenizer.tokenize(docs, min_word_count=5, stem = False, lemmatize=True, remove_stop_words=True, spelling_correct=True, number_fn= NumberStrategy.collapse_dates)
lsa_v = LatentWordVectors.LsaSpace(tokenized_docs, 100)

wds = lsa_v.word_to_index.keys()
wds = sorted(wds)

u_vecs = [ MatrixHelper.unit_vector(lsa_v.project(v)) for v in wds ]

km = cluster.KMeans(n_clusters = 50, init='k-means++', n_init=10, verbose=1, n_jobs=1)
predictions = km.fit_predict(u_vecs)

clusters = set(predictions)
word2cluster = dict(zip(wds, predictions))

km_clusters = extract_clusters_from_kmeans(km, wds)
km_clusters = sorted(km_clusters, key = lambda i: len(i))

for cl in km_clusters:
    print cl

clusters_per_doc = docs_to_clusters(tokenized_docs, word2cluster)

scores = {}
for code in data.sm_codes:
    labels = data.labels_for(code)
    
    scores_by_cluster = {}
    for cluster_id in clusters:
        in_cluster = []
        for item in clusters_per_doc:
            if cluster_id in item:
                in_cluster.append(1)
            else:
                in_cluster.append(0)
                
        f1_val = metrics.f1_score(labels, in_cluster)
        scores_by_cluster[cluster_id] = f1_val
    
    scores[code] = sort_by_value(scores_by_cluster, reverse = True)

for code in data.sm_codes:
    print code
    print scores[code][0:3]

