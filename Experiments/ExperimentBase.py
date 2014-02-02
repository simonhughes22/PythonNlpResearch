import logging
from abc import *

from CrossValidation import *
import Converter
import MatrixHelper
import Settings
import TermFrequency
import WordTokenizer
import TfIdf
import Lsa
import NumberStrategy
from Rpfa import rpfa, mean_rpfa, weighted_mean_rpfa
from ResultsHelper import dump_hits_and_misses

class ExperimentBase(object):
    __metaclass__ = ABCMeta
    MIN_DOC_LENGTH = 3
    __settings__ = Settings.Settings()

    @abstractmethod
    def sub_dir(self):
        return NotImplemented
       
    # Most experiments override these three
    @abstractmethod
    def get_vector_space(self, tokenized_docs):
        """ 
        Returns (distance_matrix, id2word)
        """
        return NotImplemented

    @abstractmethod
    def create_classifier(self, code):
        return NotImplemented
    
    # takes vector space and projects
    @abstractmethod
    def get_training_data(self, distance_matrix, id2word):
        return NotImplemented

    @abstractmethod
    def codes_to_filter(self):
        return NotImplemented

    def get_params(self):
        return ""

    # THIS ASSUMES AN SK LEARN CLASSFIER USED
    # OVERRIDE FOR NLTK
    def classify(self):
        def classify(classifier, vd):
            return classifier.predict(vd) 
        return classify

    # Common Implementations of methods above
    #VECTOR SPACE
    def term_freq_vspace(self, tokenized_docs):
        tf = TermFrequency.TermFrequency(tokenized_docs)
        return (tf.distance_matrix, tf.id2Word)        
    
    def tfidf_vspace(self, tokenized_docs):
        tfidf = TfIdf.TfIdf(tokenized_docs)
        return (tfidf.distance_matrix, tfidf.id2Word)
  
    def lsa_vspace(self, tokenized_docs):
        tfidf = TfIdf.TfIdf(tokenized_docs)
        lsa = Lsa.Lsa(tfidf, self.num_topics)
        return (lsa.distance_matrix, lsa.id2Word)
      
    #TRAINING DATA
    # For NLTK format ( "word" : freq )
    def get_binary_data(self, distance_matrix, id2word):
    # Get binary data
       return Converter.vector_space_to_dict_list(distance_matrix, id2word, Converter.to_binary)
    
    def get_sparse_matrix_data(self, distance_matrix, id2word):
        return MatrixHelper.gensim_to_numpy_array(distance_matrix, initial_value = 0)
    
    #DATA
    def get_class_value(self):
        return 1.0
    
    def get_codes(self, all_codes):
        return [c for c in all_codes
                # Exclude pure vague codes
                if c != "v" and 
                # Exclude doc codes. Need whole doc to classify them
                not c.startswith("s")]
    
    # map labels - do nothing by default
    def label_mapper(self):
        return lambda x: x

    def matrix_value_mapper(self):
        return None

    def __ensure_dir__(self, d):
        import os
        d = os.path.dirname(d)
        if not os.path.exists(d):
            os.makedirs(d)

    def __get_results_folder__(self):
        folder_for_class = str(self.__class__)[1:-2].replace("class '", "").split(".")[-1]
        results_dir = ExperimentBase.__settings__.results_directory + self.sub_dir() + "/" + folder_for_class + "/"
        return results_dir

    def __build_aggregate_results_string__(self, mean_td_metrics, mean_vd_metrics, wt_mean_td_metrics,
                                           wt_mean_vd_metrics):
        aggregate_results = "\n"
        aggregate_results += "VALIDATION DATA -\n"
        aggregate_results += "\tMEAN\n\t\t {0}\n".format(mean_vd_metrics.to_str(True))
        aggregate_results += "\tWEIGHTED MEAN\n\t\t {0}\n".format(wt_mean_vd_metrics.to_str(True))
        aggregate_results += "\n"
        aggregate_results += "TRAINING DATA -\n"
        aggregate_results += "\tMEAN\n\t\t {0}\n".format(mean_td_metrics.to_str(True))
        aggregate_results += "\tWEIGHTED MEAN\n\t\t {0}\n".format(wt_mean_td_metrics.to_str(True))
        return aggregate_results

    def __dump_results_to_file__(self, all_results, fName):
        handle = open(fName, mode="w+")
        handle.write(all_results)
        handle.close()

    def __get_labels_for_code__(self, code, data, empty_ixs, label_mapper, xs):
        ys = map(label_mapper, data.labels_for(code))
        # remove ys for empty docs
        ys = [y for i, y in enumerate(ys) if i not in empty_ixs]
        assert len(ys) == len(xs)
        return ys

    def __get_codes_to_process__(self, data, one_code):
        if one_code != None:
            codes_to_process = [one_code]
        else:
            filter_codes = self.codes_to_filter()
            codes_to_process = set(c
                                   for c in self.get_codes(data.allCodes)
                                   if filter_codes is None or c in filter_codes)

        return sorted(codes_to_process)

    def Run(self, results_file_name, cv_folds = 10, min_word_count = 5, stem = True, lemmatize = False, remove_stop_words = True, one_code = None, spelling_correct = True, one_fold = False):

        self.min_word_count = min_word_count

        #SETTINGS
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        results_dir = self.__get_results_folder__()
        self.__ensure_dir__(results_dir)

        print "Results filename: " + results_file_name
        results_file_path = results_dir + results_file_name
        vd_hits_and_misses_fname = results_file_path.replace(".txt", "_VD_hits_misses.txt")
        td_hits_and_misses_fname = results_file_path.replace(".txt", "_TD_hits_misses.txt")

        #TOKENIZE
        data = self.get_data(ExperimentBase.__settings__)
        tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count = min_word_count, stem = stem, lemmatize=lemmatize, remove_stop_words = remove_stop_words, spelling_correct=spelling_correct, number_fn = NumberStrategy.collapse_num)
    
        empty_ixs = set([i for i, doc in enumerate(tokenized_docs) if len(doc) < ExperimentBase.MIN_DOC_LENGTH])
        tokenized_docs = [t for i, t in enumerate(tokenized_docs) if i not in empty_ixs]
       
        #TRAINING DATA
        #TODO Make this one call from docs -> td
        (distance_matrix, id2word) = self.get_vector_space(tokenized_docs)
        xs = self.get_training_data(distance_matrix, id2word)
        
        matrix_mapper = self.matrix_value_mapper()
        if matrix_mapper:
            xs = MatrixHelper.map_matrix(matrix_mapper, xs)
        
        all_results = self.get_params() + "\n"
        print all_results,
        
        MIN_CODE_COUNT = 1
        
        vd_metrics, td_metrics = [], []
        label_mapper = self.label_mapper()

        # Stop logging now
        logging.disable(logging.INFO)
        
        # So we can test on one code only
        codes_to_process = self.__get_codes_to_process__(data, one_code)

        # Store the indices into the inputs that detail
        # the true and false positives and negatives
        vd_hits_misses_by_code = dict()
        td_hits_misses_by_code = dict()

        for code in codes_to_process:

            ys = self.__get_labels_for_code__(code, data, empty_ixs, label_mapper, xs)
                  
            total_codes = len([item for item in ys if item == 1])
            if total_codes <= MIN_CODE_COUNT:
                continue

            # Yes, that is a lot I know
            vd_r, vd_p, vd_f1, vd_a, \
            td_r, td_p, td_f1, td_a, \
            vd_tp_ix, vd_fp_ix, vd_fn_ix, vd_tn_ix, \
            td_tp_ix, td_fp_ix, td_fn_ix, td_tn_ix \
                = cross_validation_score_generic(
                    xs, ys,
                    self.create_classifier(code),
                    self.classify(),
                    cv_folds,
                    class_value = self.get_class_value(),
                    one_fold = one_fold)

            vd_metric, td_metric = rpfa(vd_r, vd_p, vd_f1, vd_a, total_codes), rpfa(td_r, td_p, td_f1, td_a, total_codes)
            vd_metrics.append(vd_metric)
            td_metrics.append(td_metric)

            vd_hits_misses_by_code[code] = (vd_tp_ix, vd_fp_ix, vd_fn_ix, vd_tn_ix)
            td_hits_misses_by_code[code] = (td_tp_ix, td_fp_ix, td_fn_ix, td_tn_ix)

            results = "Code: {0} Count: {1} VD[ {2} ]\tTD[ {3} ]\n".format(code.ljust(7), str(total_codes).rjust(4), vd_metric.to_str(), td_metric.to_str())
            print results,
            all_results += results

            """ Dump results to file in case of crash """
            self.__dump_results_to_file__(all_results, results_file_path)
            dump_hits_and_misses(vd_hits_misses_by_code, xs, vd_hits_and_misses_fname)
            dump_hits_and_misses(td_hits_misses_by_code, xs, td_hits_and_misses_fname)

        """ Compute mean metrics """
        """ MEAN """
        mean_vd_metrics,    mean_td_metrics     = mean_rpfa(vd_metrics),           mean_rpfa(td_metrics)
        """ WEIGHTED MEAN """
        wt_mean_vd_metrics, wt_mean_td_metrics  = weighted_mean_rpfa(vd_metrics),  weighted_mean_rpfa(td_metrics)

        str_aggregate_results = self.__build_aggregate_results_string__(mean_td_metrics, mean_vd_metrics,
                                                                    wt_mean_td_metrics, wt_mean_vd_metrics)
        print str_aggregate_results
        all_results += str_aggregate_results
            
        #DUMP TO FILE
        print "Writing results to: " + results_file_path
        print "TD Hits and Misses: " + td_hits_and_misses_fname
        print "VD Hits and Misses: " + vd_hits_and_misses_fname

        self.__dump_results_to_file__(all_results, results_file_path)
        return (mean_vd_metrics, wt_mean_vd_metrics)