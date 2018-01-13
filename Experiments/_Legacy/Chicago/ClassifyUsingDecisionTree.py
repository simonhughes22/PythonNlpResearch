import Converter
import SentenceData
import DictionaryHelper
import DocumentFrequency
import Settings
import TermFrequency
import WordTokenizer
import datetime
import logging
import nltk
    
def to_binary(item):
        if item > 0:
            return 1
        return 0

def extract_best_n_words(relative_freq, n_words, xs):
    first_n_words = set(
                        [item[0] 
                         for item in DictionaryHelper.sort_by_value(relative_freq, reverse = True)]
                        [:n_words]
                        )
    
    condensed_data = []
    for row in xs:
        new_row = dict()
        condensed_data.append(new_row)
        for item in row.items():
            if item[0] in first_n_words:
                new_row[item[0]] = item[1]
                
    return condensed_data

def train():

    #SETTINGS
    best_n_words = 10000
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    settings = Settings.Settings()
    results_dir = settings.results_directory + "Chicago/Files"

    #TOKENIZE
    xs = SentenceData.SentenceData()
    tokenizer = WordTokenizer.WordTokenizer(min_word_count = 5)
    tokenized_docs = tokenizer.tokenize(xs.documents)

    term_freq = TermFrequency.TermFrequency(tokenized_docs)
    
    #NLTK Decision Tree
    list_of_dicts = Converter.vector_space_to_dict_list(term_freq.distance_matrix, term_freq.id2Word, to_binary)
    
    #Number of unique documents
    unique_doc_count = len(xs.documents)
    
    for smCode in xs.sm_codes[1:]: # Skip the first (A code)
        try:
            
            print "{0} - Processing smCode: {1}".format(str(datetime.datetime.now()), smCode) 
            
            labels = xs.labels_for(smCode)
            num_docs_for_code = len(xs.sentences_for_code(smCode))
            print "#Docs for code: {0}".format(num_docs_for_code)
                
            relative_word_frequency = DocumentFrequency.document_frequency_ratio(list_of_dicts, labels, lambda l: l == 1)
            condensed_data = extract_best_n_words(relative_word_frequency, best_n_words, list_of_dicts)

            training_data = zip(condensed_data, labels)
        
            dt = nltk.DecisionTreeClassifier.train(training_data)
            
            results = "Num Words Used: " + str(best_n_words) + "\n"
            results += smCode + "\n\n"
            results += "PSEUDOCODE:\n"
            results += dt.pseudocode(depth = 1000) + "\n"
            
            error = dt.error(training_data)
        
            results += "ERROR:         " + str(error * 100) + "%\n\n"
            
            pct_docs_for_code = num_docs_for_code / float(unique_doc_count) * 100.0
            results += "Docs for code: {0}%\n".format(pct_docs_for_code) 
            
            fName = results_dir + smCode + ".txt"

            handle = open(fName, mode = "w+")
            handle.write(results)
            handle.close()
            print results
        
        except IOError as e:
            print str(e)

    #binary_matrix = term_freq.binary_matrix()
    #decision_tree = tree.DecisionTreeClassifier(criterion = 'entropy')
    #decision_tree.fit(binary_matrix, labels)
    
    # Test with CL1 labels
    raw_input("Press Enter to quit")
    

if __name__ == "__main__":
    train()