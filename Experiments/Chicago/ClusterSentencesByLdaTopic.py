import ClustersToFile
import SentenceData
import Lda
import TfIdf
import WordTokenizer as dbnetwork
import logging

def extract_topic_labels(distance_matrix):
    return [ [] if len(row) == 0 else [tpl[0] for tpl in row] for row in distance_matrix ]


def train(num_lda_topics):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s :     %(message)s', level=logging.INFO)

    #TOKENIZE
    xs = SentenceData.SentenceData()
    
    tokenizer = dbnetwork.WordTokenizer(min_word_count = 5)
    tokenized_docs = tokenizer.tokenize(xs.documents)

    #MAP TO VECTOR AND SEMANTIC SPACE
    tfidf = TfIdf.TfIdf(tokenized_docs)
    lda = Lda.Lda(tfidf, num_topics = num_lda_topics)

    # Pull out topic topic_labels    
    topic_labels = extract_topic_labels(lda.distance_matrix)
    
    #OUTPUT

    file_name_code_clusters = "LDA_SMCODES_topics_{0}.csv".format(num_lda_topics)
    ClustersToFile.clusters_to_file(file_name_code_clusters, topic_labels, xs.codes_per_document, "Chicago")
    
    file_name_category_clusters = "LDA_categories_topics_{0}.csv".format(num_lda_topics)
    ClustersToFile.clusters_to_file(file_name_category_clusters, topic_labels, xs.categories_per_document, "Chicago")
    
    print "Finished processing lda clustering for dims: {0}".format(num_lda_topics)

    
if __name__ == "__main__":
    
    train(num_lda_topics = 30)