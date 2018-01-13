from collections import defaultdict
import logging

import Clusterer
import ClustersToFile
import SentenceData
import Lsa
import MatrixHelper
import TfIdf
import WindowSplitter
import WordTokenizer as dbnetwork


def split_documents_into_windows(lst_tokens, window_size):
    """ Takes a list of tokens and turns them into a larger list of  
        windows of tokens of a fixed size (or less).
        Returns a tuple of two lists, the first the list of windows
        and the second a list of indices into the original list of tokens
    """
    
    windowed_tokens = []
    indices = []
    for i, lst in enumerate(lst_tokens):
        windows = WindowSplitter.split_into_windows(lst, window_size)
        indices.extend([i for w in windows])
        windowed_tokens.extend(windows)
        
    return (windowed_tokens,  indices)

def pivot_window_labels(window_cluster_labels, window_indices):
    d = defaultdict(set)
    for i, index in enumerate(window_indices):
        d[index].add(window_cluster_labels[i])
    
    return [list(labels) 
            # Ensure dict is sorted by index 
            for index,labels in sorted(d.items(), key = lambda item: item[0])]
   
   
# ****** TODO DOES THIS WORK !!! ******
def train(num_lsa_topics, k, window_size):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #TOKENIZE
    xs = SentenceData.SentenceData()
    
    tokenizer = dbnetwork.WordTokenizer(min_word_count = 5)
    tokenized_docs = tokenizer.tokenize(xs.documents)
    windowed_docs, window_indices = split_documents_into_windows(tokenized_docs, window_size)
    
    #MAP TO VECTOR AND SEMANTIC SPACE
    tfidf = TfIdf.TfIdf(windowed_docs)
    lsa = Lsa.Lsa(tfidf, num_topics = num_lsa_topics)
    full_lsa_matrix = MatrixHelper.gensim_to_python_mdarray(lsa.distance_matrix, num_lsa_topics)

    #CLUSTER
    clusterer = Clusterer.Clusterer(k)
    window_labels = clusterer.Run(full_lsa_matrix)
    
    #Extract the labeld for the original sentences using the indices build earlier
    labels = pivot_window_labels(window_labels, window_indices)

    #OUTPUT
    file_name_code_clusters = "Windowed_LSA_SMCODES_win_size_{0}_k-means_k_{1}_dims_{2}.csv".format(window_size, k, num_lsa_topics)
    ClustersToFile.clusters_to_file(file_name_code_clusters, labels, xs.codes_per_document, "Chicago")
    
    file_name_category_clusters = "Windowed_LSA_Categories_win_size_{0}_k-means_k_{1}_dims_{2}.csv".format(window_size, k, num_lsa_topics)
    ClustersToFile.clusters_to_file(file_name_category_clusters, labels, xs.categories_per_document, "Chicago")
    
    logging.info("Finished processing lsa clustering for dims: {0} and k: {1}".format(num_lsa_topics, k))

    
if __name__ == "__main__":
  
    #k = cluster size
    #for k in range(40,41,1): #start, end, increment size
    #    train(300, k)

    train(num_lsa_topics = 300, k = 30 , window_size = 7)