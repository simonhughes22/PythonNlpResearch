import Clusterer
import ClustersToFile
import SentenceData
import Lsa
import MatrixHelper
import TfIdf
import WordTokenizer
import logging

def train(num_lsa_topics, k):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    #TOKENIZE
    xs = SentenceData.SentenceData()
    
    tokenizer = WordTokenizer.WordTokenizer(min_word_count = 5)
    tokenized_docs = tokenizer.tokenize(xs.documents)

    #MAP TO VECTOR AND SEMANTIC SPACE
    tfidf = TfIdf.TfIdf(tokenized_docs)
    lsa = Lsa.Lsa(tfidf, num_topics = num_lsa_topics)
    full_lsa_matrix = MatrixHelper.gensim_to_python_mdarray(lsa.distance_matrix, num_lsa_topics)

    #CLUSTER
    clusterer = Clusterer.Clusterer(k)
    labels = clusterer.Run(full_lsa_matrix)

    #OUTPUT
    file_name_code_clusters = "LSA_SMCODES_k-means_k_{0}_dims_{1}.csv".format(k, num_lsa_topics)
    ClustersToFile.clusters_to_file(file_name_code_clusters, labels, xs.codes_per_document, "Chicago")
    
    file_name_category_clusters = "LSA_categories_k-means_k_{0}_dims_{1}.csv".format(k, num_lsa_topics)
    ClustersToFile.clusters_to_file(file_name_category_clusters, labels, xs.categories_per_document, "Chicago")
    
    print "Finished processing lsa clustering for dims: {0} and k: {1}".format(num_lsa_topics, k)

    
if __name__ == "__main__":
  
    #k = cluster size
    #for k in range(40,41,1): #start, end, increment size
    #    train(300, k)

    train(num_lsa_topics = 300, k = 30 )