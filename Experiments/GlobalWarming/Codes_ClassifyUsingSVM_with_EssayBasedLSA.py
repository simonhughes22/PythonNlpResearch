from CrossValidation import cross_validation, cross_validation_score
from Labeler import *
from numpy import *
from sklearn import svm
import Converter
import GwData
import Lsa
import MatrixHelper
import Metrics
import ResultsHelper
import Settings
import TermFrequency
import TfIdf

import WordTokenizer
import logging
import nltk

def train():

    #SETTINGS
    cv_folds = 10
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    settings = Settings.Settings()
    results_dir = settings.results_directory + + GwData.FOLDER
    num_lsa_topics = 100

    #TOKENIZE
    xs = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(xs.documents, min_word_count = 5)
    tfidf = TfIdf.TfIdf(tokenized_docs)
    lsa = Lsa.Lsa(tfidf, num_topics = num_lsa_topics)

    #NLTK SVM linear kernel
    xs = MatrixHelper.gensim_to_numpy_array(lsa.distance_matrix, initial_value = 0)

    total_recall, total_precision, total_f1 = 0.0, 0.0, 0.0

    all_results = "LSA Dimensions: " + str(num_lsa_topics)
    print all_results

    processed_code_count = 0
    #MIN_CODE_COUNT = 5

    MIN_CODE_COUNT = 1

    codes = [c for c in xs.sm_codes
             # Exclude pure vague codes
             if c != "v" and
             # Exclude doc codes. Need whole doc to classify them
             not c.startswith("s")]

    for code in codes:

        code_count = xs.sm_code_count[code]
        if code_count <= MIN_CODE_COUNT:
            continue

        processed_code_count += 1
        labels = map(Converter.get_svm_val , xs.labels_for(code))
        classifier = svm.LinearSVC(C = 1)
        recall, precision, f1_score = cross_validation_score(xs, labels, classifier, cv_folds, class_value = 1.0)
        results = "Code: {0} Count: {1}, Recall: {2}, Precision: {3}, F1: {4}\n".format(code.ljust(10), code_count, recall, precision, f1_score)

        all_results += results
        total_recall += recall
        total_precision += precision
        total_f1 += f1_score

        print results,

    #num_codes = len(xs.sm_codes)
    num_codes = processed_code_count
    result = "AGGREGATE\n\t Recall: {0}, Precision: {1}, F1: {2}\n".format(total_recall / num_codes, total_precision / num_codes, total_f1 / num_codes)
    all_results += result
    print result

    #DUMP TO FILE
    fName = results_dir + "Codes_ClassifyUsing_SVM_with_EssayBasedLSA_Dims_" + str(num_lsa_topics) + ".txt"
    handle = open(fName, mode = "w+")
    handle.write(all_results)
    handle.close()

    #raw_input("Press Enter to quit")

if __name__ == "__main__":
    train()