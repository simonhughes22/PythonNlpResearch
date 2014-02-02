from Labeler import *
import Converter
import GwData
import ResultsHelper
import Settings
import TermFrequency
import WordTokenizer
import logging
import nltk


def main():

    #SETTINGS
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    settings = Settings.Settings()
    results_dir = settings.results_directory + "GlobalWarming\\"

    #TOKENIZE
    data = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count = 5)
    term_freq = TermFrequency.TermFrequency(tokenized_docs)
    
    #NB
    list_of_dicts = Converter.vector_space_to_dict_list(term_freq.distance_matrix, term_freq.id2Word, Converter.to_binary)
    
    labels = data.causal_per_document
    labelled_data = zip(list_of_dicts, labels)
    td_size = int(0.75 * len(labelled_data))
    
    training_data = labelled_data[:td_size]
    validation_data = labelled_data[td_size:]

    nb = nltk.NaiveBayesClassifier.train(training_data)
    
    #RESULTS
    classifications = [nb.classify(rcd) for rcd,lbl in validation_data]
    
    results = ResultsHelper.rfp(labels[td_size:], classifications)

    results += "(100) MOST INFORMATIVE FEATURES:\n"
    features = nb.most_informative_features(100)
    for i,(f,val) in enumerate(features):
        results += "\t" + str(i + 1) + " : " + f + " -> " + str(val) + "\n"

    print results

    #DUMP TO FILE    
    fName = results_dir + "Causal_Relation_NB.txt"
    handle = open(fName, mode = "w+")
    handle.write(results)
    handle.close()
    
    #binary_matrix = term_freq.binary_matrix()
    #decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy')
    #decision_tree.fit(binary_matrix, labels)
    
    # Test with CL1 labels
    raw_input("Press Enter to quit")
    

if __name__ == "__main__":
    main()