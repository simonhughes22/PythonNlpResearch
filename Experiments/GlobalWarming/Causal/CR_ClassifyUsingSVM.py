from Labeler import *
from sklearn import svm
import GwData
import MatrixHelper
import ResultsHelper
import Settings
import TermFrequency
import TfIdf
import WordTokenizer
import logging
import nltk
from numpy import *

def main():

    #SETTINGS
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    settings = Settings.Settings()
    results_dir = settings.results_directory + "GlobalWarming\\"

    #TOKENIZE
    data = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count = 5)
    tfidf = TfIdf.TfIdf(tokenized_docs)
    
    #NLTK Decision Tree
    np_matrix = MatrixHelper.gensim_to_numpy_array(tfidf.matrix, initial_value = 0)
    
    labels = data.causal_per_document
    
    def get_svm_val(x):
        if x <= 0:
            return -1
        return 1

    labels = map(get_svm_val,labels)

    td_size = int(0.75 * len(np_matrix))
    
    td_x = np_matrix[:td_size]
    td_y = labels[:td_size]
    
    vd_x = np_matrix[td_size:]
    vd_y = labels[td_size:]
    
    rng = array(range(1,21,1))
    
    c_vals = rng / 10.0
    
    all_results = ""
    for c in c_vals:
        classifier = svm.LinearSVC(C = c)
        classifier.fit(td_x, td_y)
        
        #RESULTS
        classifications = classifier.predict(vd_x)
        
        results = "\nC VALUE: " + str(c) + "\n"
        results += ResultsHelper.rfp(vd_y, classifications)
        print results

        all_results += results
    #print "EXPLAIN:\n"
    #me.explain(condensed_data[0], 100)

    #DUMP TO FILE    
    fName = results_dir + "Causal_Relation_SVM.txt"
    handle = open(fName, mode = "w+")
    handle.write(all_results)
    handle.close()
    
    #binary_matrix = term_freq.binary_matrix()
    #decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy')
    #decision_tree.fit(binary_matrix, labels)
    
    # Test with CL1 labels
    raw_input("Press Enter to quit")

if __name__ == "__main__":
    main()