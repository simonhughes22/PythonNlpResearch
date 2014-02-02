import Clusterer
import ClustersToFile
import SentenceData
import ListHelper
import Lsa
import MatrixHelper
import TfIdf
import WordTokenizer
import logging
import PartitionByCode
import CosineSimilarity
import collections

def find_closest_document(txtMatrixByCode, row):
    """ Takes a dictionary of codes to LSA matrices (one per document)
        and returns the key for the closest doc based on the mean
        cosine similarity (could also use max...)
    """
    if len(row) == 0:
        return "ERROR"
    
    means_per_code = {}
    
    for doc in txtMatrixByCode.keys():
        distance_matrix = txtMatrixByCode[doc]
        total = 0.0
        for row_to_test in distance_matrix:
            sim = CosineSimilarity.cosine_similarity(row, row_to_test)
            total += sim
        means_per_code[doc] = total / len(distance_matrix)
        
    # first row, first tuple (key)
    return sorted(means_per_code.items(), key = lambda item: item[1], reverse = True)[0][0] 
    
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

    #TODO Partition into Docs by LSA sim
    txt_codes = xs.text_codes
    clusters_per_text_code = int(round( k/ float((len(txt_codes)))))
    
    #Extract the sm code rows from LSA
    smCodeRows = ListHelper.filter_list_by_index(full_lsa_matrix, xs.sm_code_indices)
    smCodeClassifications = ListHelper.filter_list_by_index(xs.codes_per_document, xs.sm_code_indices)
    smCodeCategoryClassifications = ListHelper.filter_list_by_index(xs.categories_per_document, xs.sm_code_indices)
    
    # Dict of <code, list[list]]> - LSA row vectors
    logging.info("Partitioning LSA distance_matrix by Source Document")
    
    txtMatrixByCode = PartitionByCode.partition(full_lsa_matrix, xs, xs.text_codes)
    closest_docs = [find_closest_document(txtMatrixByCode, row) for row in smCodeRows]
    matrix_by_doc = collections.defaultdict(list)
    
    for i, doc in enumerate(closest_docs):
        matrix_by_doc[doc].append(smCodeRows[i])

    #Stores all cluster labels
    logging.info("Clustering within a document")
    all_smcode_labels = []
    label_offset = 0
    for doc in xs.text_codes:
        distance_matrix = matrix_by_doc[doc]
        #CLUSTER
        clusterer = Clusterer.Clusterer(clusters_per_text_code)
        labels = clusterer.Run(distance_matrix)
        all_smcode_labels = all_smcode_labels + [int(l + label_offset) for l in labels]
        label_offset += clusters_per_text_code

    #OUTPUT
    file_name_code_clusters = "Partition_By_Doc_LSA_SMCODES_k-means_k_{0}_dims_{1}.csv".format(k, num_lsa_topics)
    ClustersToFile.clusters_to_file(file_name_code_clusters, all_smcode_labels, smCodeClassifications, "Chicago")

    file_name_category_clusters = "Partition_By_Doc_LSA_categories_k-means_k_{0}_dims_{1}.csv".format(k, num_lsa_topics)
    ClustersToFile.clusters_to_file(file_name_category_clusters, all_smcode_labels, smCodeCategoryClassifications, "Chicago")
    
    #TODO - filter the category and the docs per docs to the sm codes and output
    #file_name_category_clusters = "Partition_By_Doc_LSA_categories_k-means_k_{0}_dims_{1}.txt".format(k, num_lsa_topics)
    #ClustersToFile.clusters_to_file(file_name_category_clusters, all_smcode_labels, smCodeClassifications, "Chicago")

    print "Finished processing lsa clustering for dims: {0} and k: {1}".format(num_lsa_topics, k)

    
if __name__ == "__main__":
   
    #k = cluster size
    #for k in range(40,41,1): #start, end, increment size
    #    train(300, k)

    train(num_lsa_topics = 300, k = 30)