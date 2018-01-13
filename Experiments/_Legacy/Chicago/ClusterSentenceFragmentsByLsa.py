import Clusterer
import ClustersToFile
import SentenceFragmentData
import Lsa
import MatrixHelper
import TfIdf
import WordTokenizer
import logging
import ListHelper

def train(num_lsa_topics, k):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #TOKENIZE
    xs = SentenceFragmentData.SentenceFragmentData()
    
    tokenizer = WordTokenizer.WordTokenizer(min_word_count = 5)
    tokenized_docs = tokenizer.tokenize(xs.documents)

    #MAP TO VECTOR AND SEMANTIC SPACE
    tfidf = TfIdf.TfIdf(tokenized_docs)
    lsa = Lsa.Lsa(tfidf, num_topics = num_lsa_topics)
    full_lsa_matrix = MatrixHelper.gensim_to_python_mdarray(lsa.distance_matrix, num_lsa_topics)
    
    #Filter To just sm codes
    sm_code_lsa_matrix = ListHelper.filter_list_by_index(full_lsa_matrix, xs.sm_code_indices)
    
    #CLUSTER
    clusterer = Clusterer.Clusterer(k)
    labels = clusterer.Run(sm_code_lsa_matrix)

    #OUTPUT - Filter by SM Code only this time
    file_name_code_clusters = "LSA_SMCODES_Fragments_k-means_k_{0}_dims_{1}.csv".format(k, num_lsa_topics)
    sm_codes_per_doc   = ListHelper.filter_list_by_index(xs.codes_per_document, xs.sm_code_indices)
    ClustersToFile.clusters_to_file(file_name_code_clusters, labels, sm_codes_per_doc, "Chicago")
    
    file_name_category_clusters = "LSA_Categories_Fragments_k-means_k_{0}_dims_{1}.csv".format(k, num_lsa_topics)
    categories_per_doc = ListHelper.filter_list_by_index(xs.categories_per_document, xs.sm_code_indices)
    ClustersToFile.clusters_to_file(file_name_category_clusters, labels, categories_per_doc, "Chicago")
    
    print "Finished processing lsa clustering for dims: {0} and k: {1}".format(num_lsa_topics, k)

    
if __name__ == "__main__":
  
    #k = cluster size
    #for k in range(40,41,1): #start, end, increment size
    #    train(300, k)

    train(num_lsa_topics = 300, k = 100 )