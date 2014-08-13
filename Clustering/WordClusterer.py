from collections import defaultdict

from gensim import similarities
from sklearn import cluster
import numpy as np
from nltk.corpus import stopwords

from LatentWordVectors import LatentWordVectors
from MatrixHelper import numpy_to_gensim_format


class WordCluster(object):
    
    def __init__(self, tokenized_docs, num_topics = 100, min_doc_freq = 5, projector_fn = None, aggregation_method="doc"):

        """ PRE PROCESSING """
        """ Temporary functions """
        def compute_doc_freq(tkns):
            tally = defaultdict(int)
            for doc in map(set, tkns):
                for t in doc:
                    tally[t] += 1
            return tally

        def remove_less_than(tkns, tally, min_cnt):
            def lt(tkns):
                return [t for t in tkns if tally[t] >= min_cnt]

            return map(lt, tkns)

        def remove_stop_words(tokenized_docs, stop_words):
            return [[w for w in doc if w not in stop_words] for doc in tokenized_docs ]

        """ End temporary functions """

        self.docs = tokenized_docs
        self.doc_freq = compute_doc_freq(tokenized_docs)
        if min_doc_freq > 1:
            self.docs = remove_less_than(tokenized_docs, self.doc_freq, min_doc_freq)

        if remove_stop_words:
            stop_wds = stopwords.extract_words("english")
            self.docs = remove_stop_words(self.docs, stop_wds)

        self.num_unique_tokens = self.__count_unique_tokens__(self.docs)

        """ END PRE PROCESSING """

        self.num_topics = num_topics
        self.min_doc_freq = min_doc_freq

        if projector_fn is None:
            projector_fn = LatentWordVectors.LsaTfIdfSpace

        # Note the gensim TfIdf impl normalizes
        self.projector = projector_fn(self.docs, num_topics=num_topics, aggregation_method=aggregation_method, normalize=False, unit_vectors=True)

        self.words = self.projector.word_to_index.keys()
        self.word2index = {}

        ix = 0
        for w in self.words:
            self.word2index[w] = ix
            ix += 1

        self.projections = np.array([self.projector.project(w) for w in self.words])
        self.sims = None

        # Can take a while, so report completion
        print("WordClusterer Created")
        pass

    def __unique_tokens__(self, token_groups):
        
        tkns = set()
        for grp in token_groups:
            for token in grp:
                tkns.add(token)
        return tkns

    def __count_unique_tokens__(self, token_groups):
        return len(self.__unique_tokens__(token_groups))

    def top_n_words(self, n=None):

        if n is None:
            # all words
            n = len(self.words)

        if self.sims == None:
            gs_projections = numpy_to_gensim_format(self.projections)
            mat_sim = similarities.MatrixSimilarity(corpus=gs_projections, num_features=self.num_topics)
            self.sims = mat_sim[gs_projections]
      
        return self.__get_top_n__(self.sims, self.words, self.words, n)

    def print_top_n_words(self, n):
        top_n = self.top_n_words(n)
        self.__print_top_n__(top_n)


    def __print_top_n__(self, top_n):
        s_top_n = sorted(top_n, key=lambda (w, lst): w)
        for a, lst in s_top_n:
            print a
            for b, sim in lst:
                print "\t" + b.ljust(35) + "\t" + str(sim)
            print ""

    def list_similar_words(self, simlarity_threshold = 0.0):
        """
            List all terms in the object and all other terms that
            have a similarity at or above the simlarity_threshold.
            Returns a list of (word, [(word, sim), (word, sim), ...])
        """
        gs_projections = numpy_to_gensim_format(self.projections)
        mat_sim = similarities.MatrixSimilarity(corpus=gs_projections, num_features=self.num_topics)
        sims = mat_sim[gs_projections]
        return self.__get_similar_words__(sims, self.words, simlarity_threshold)

    def __get_similar_words__(self, sims, items, simlarity_threshold):
        items = np.array(items)
        s_sims = np.argsort(sims)

        d_sim_words = {}
        for i, (current_word, sim_indexs) in enumerate(zip(items, s_sims)):
            #Reverse the indices
            matches = []
            r_indexes = sim_indexs[::-1]
            for ix in r_indexes:
                sim = sims[i, ix]
                if sim < simlarity_threshold:
                    break

                similar_word = items[ix]
                if current_word == similar_word:
                   continue

                matches.append((similar_word, sim))
            d_sim_words[current_word] =  matches

        swords = sorted(d_sim_words.keys())

        """ Return output sorted by word order (alpha) """
        lst_sim_words = []
        for w in swords:
            matches = d_sim_words[w]
            lst_sim_words.append((w, matches))
        return lst_sim_words

    def __get_top_n__(self, sims, subset_items, all_items, n):
        all_items = np.array(all_items)
        s_sims = np.argsort(sims)

        top_n_items = all_items[s_sims[:, -n:-1]]

        def reverse(a):
            return a[::-1]
        top_n_items = map(reverse, top_n_items)
        top_n = zip(subset_items, top_n_items)
        top_n = map(lambda (w, lst): (w, map( lambda w2: (w2, sims[ self.word2index[w], self.word2index[w2] ]) ,lst )), top_n)
        return sorted(top_n, key = lambda tpl: tpl[0])

    def density_cluster_by_sim(self, sim_threshold = 0.7):
    
        item2cluster = {}
        density_clusters = []
        
        top_100_pairs = self.top_n_words(100)
        
        for a, lst in top_100_pairs:
            for b,sim in lst:
                if sim < sim_threshold:
                    break
                if b in item2cluster and a in item2cluster:
                    continue
                
                if b in item2cluster:
                    ix = item2cluster[b]
                    density_clusters[ix].append(a)
                    item2cluster[a] = ix
                elif a in item2cluster:
                    ix = item2cluster[a]
                    density_clusters[ix].append(b)
                    item2cluster[b] = ix
                else:
                    ix = len(density_clusters)
                    
                    l = [a,b]
                    density_clusters.append(l)
                    
                    item2cluster[b] = ix
                    item2cluster[a] = ix
        
        density_clusters = map(sorted, density_clusters)
        return sorted(density_clusters, key = lambda cl: len(cl))

    def print_density_clusters(self, sim_threshold = 0.7):
        clusters = self.density_cluster_by_sim(sim_threshold)
        self.print_clusters(clusters)

    def k_means_clusters(self, k = 100):
        km = cluster.KMeans(n_clusters = k, init='k-means++', n_init=10, verbose=1, n_jobs=1)
        """ Takes a while """
        km.fit(self.projections)
        km_clusters = self.__extract_clusters_from_clusterer__(km, self.words)
        return km_clusters

    def print_k_means_clusters(self, k = 100):
        clusters = self.k_means_clusters(k)
        self.print_clusters(clusters)
    
    def dbscan_clusters(self, max_distance = 0.2):
        
        dbscan = cluster.DBSCAN(eps = max_distance, metric = 'precomputed')
        dist_matrix = 1.0 - ((1.0 + np.dot(self.projections, self.projections.T)) / 2.0)
        """ Remove very small values < 0 due to rounding errors """
        lt = -1.0 * (dist_matrix * (dist_matrix < 0.0))
        dist_matrix = dist_matrix + lt
        
        """ Cluster """
        dbscan.fit(dist_matrix)
        dbscan_clusters = self.__extract_clusters_from_clusterer__(dbscan, self.words)
        return dbscan_clusters
    
    def print_dbscan_clusters(self, max_distance = 0.2):
        cl = self.dbscan_clusters(max_distance)
        self.print_clusters(cl)
    
    def __extract_clusters_from_clusterer__(self, clusterer, items):
        clusters = defaultdict(list)
        for i,lbl in enumerate(clusterer.labels_):
            """ DBscan assigns a -1 to noisy samples """
            if lbl >= 0:
                clusters[lbl].append(items[i])
        return map(sorted, clusters.values())

    def print_clusters(self, clusters):
        for cl in clusters:
            print cl

    def unclustered_items(self, clusters):
        cluster_tokens = self.__unique_tokens__(clusters)
        missing = []
        for t in self.words:
            if t not in cluster_tokens:
                missing.append(t)
        return missing
    
    def print_cluster_info(self, clusters):
        token_cnt = self.__count_unique_tokens__(clusters)
        mean_size = np.mean( map(len, clusters))
        pct_coverage = token_cnt / float(self.num_unique_tokens) * 100
        print("{0} tokens out of {1}\n% coverage: {2}\nNum Clusters: {3}\nMean Size: {4}".format(token_cnt, self.num_unique_tokens, pct_coverage, len(clusters),mean_size))
