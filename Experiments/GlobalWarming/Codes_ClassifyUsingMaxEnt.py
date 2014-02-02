from CrossValidation import *
import Converter
import GwData
import Settings
import TfIdf
import WordTokenizer
import logging
import nltk
import Lsa
import TermFrequency
import MatrixHelper
from Labeler import *
import ResultsHelper

from collections import defaultdict

def train():

    #SETTINGS
    cv_folds = 10
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    settings = Settings.Settings()
    results_dir = settings.results_directory + GwData.FOLDER

    #TOKENIZE
    xs = GwData.GwData()
    tokenized_docs = WordTokenizer.tokenize(xs.documents, min_word_count = 5)

    tfidf = TfIdf.TfIdf(tokenized_docs)
    lsa = Lsa.Lsa(tfidf, 300)

    #NLTK SVM linear kernel
    xs = Converter.vector_space_to_dict_list(lsa.distance_matrix, lsa.id2Word, fn_dict_creator = lambda : defaultdict(float))
    total_recall, total_precision, total_f1, total_accuracy = 0.0, 0.0, 0.0, 0.0

    all_results = ""
    processed_code_count = 0

    def create_maxent(xs, ys):
        td = zip(xs, ys)
        return nltk.MaxentClassifier.train(td, algorithm='GIS', trace = 2, max_iter = 10)

    def classify(me, xs):
        return me.batch_classify(xs)

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
        labels = xs.labels_for(code)

        recall, precision, f1_score, accuracy = cross_validation_score_generic(xs, labels, create_maxent, classify, cv_folds, class_value = 1.0)
        results = "Code: {0} Count: {1}, Recall: {2}, Precision: {3}, F1: {4}\n".format(code.ljust(10), code_count, recall, precision, f1_score)

        all_results += results
        total_recall += recall
        total_precision += precision
        total_f1 += f1_score
        total_accuracy += accuracy

        print results,

    #num_codes = len(xs.sm_codes)
    num_codes = processed_code_count
    result = "AGGREGATE\n\t Recall: {0}, Precision: {1}, F1: {2}, Accuracy {3}\n".format(total_recall / num_codes, total_precision / num_codes, total_f1 / num_codes, total_accuracy / num_codes)
    all_results += result
    print result

    #DUMP TO FILE
    fName = results_dir + "Code_Classify_MaxEnt.txt"
    handle = open(fName, mode = "w+")
    handle.write(all_results)
    handle.close()

    raw_input("Press Enter to quit")

if __name__ == "__main__":
    train()
