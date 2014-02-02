import Converter
import GwData
import DictionaryHelper
import DocumentFrequency
import Settings
import TermFrequency
import WordTokenizer
import datetime
import logging
import nltk
import ResultsHelper

from Labeler import *

def extract_best_n_words(relative_freq, n_words, data):
    first_n_words = set(
                        [item[0] 
                         for item in DictionaryHelper.sort_by_value(relative_freq, reverse = True)]
                        [:n_words]
                        )
    
    condensed_data = []
    for row in data:
        new_row = dict()
        condensed_data.append(new_row)
        for item in row.items():
            if item[0] in first_n_words:
                new_row[item[0]] = item[1]
                
    return condensed_data

def main():

    #SETTINGS
    best_n_words = 10000
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    settings = Settings.Settings()
    results_dir = settings.results_directory + "GlobalWarming\\"

    #TOKENIZE
    data = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count = 5)
    term_freq = TermFrequency.TermFrequency(tokenized_docs)
    
    #NLTK Decision Tree
    list_of_dicts = Converter.vector_space_to_dict_list(term_freq.matrix, term_freq.id2Word, Converter.to_binary)
 
    labels = data.causal_per_document
    causal_count = sum(labels)
    relative_word_frequency = DocumentFrequency.document_frequency_ratio(list_of_dicts, labels, lambda l: l == 1)
    condensed_data = extract_best_n_words(relative_word_frequency, best_n_words, list_of_dicts)

    labelled_data = zip(condensed_data, labels)
    td_size = int(0.75 * len(labelled_data))
    
    training_data = labelled_data[:td_size]
    validation_data = labelled_data[td_size:]

    dt = nltk.DecisionTreeClassifier.train(training_data)
    
    #RESULTS
    classifications = [dt.classify(rcd) for rcd,lbl in validation_data]
    
    results = ResultsHelper.rfp(labels[td_size:], classifications)
    results +=  "Num Words Used              : " + str(best_n_words) + "\n"
    results += "\n"

    error = dt.error(labelled_data)
    results += "ERROR:                      : " + str(error * 100) + "%\n"
    results += "\n"

    results += "PSEUDOCODE:\n"
    results += dt.pseudocode(depth = 1000) + "\n"
    print results

    #DUMP TO FILE    
    fName = results_dir + "Causal_Relation_DT.txt"
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