'''
Created on Apr 14, 2013
@author: Simon
'''
from collections import defaultdict

from CrossValidation import cross_validation_edges
import Metrics
import numpy as np
import logging
import MatrixHelper
import Settings
import WordTokenizer
import NumberStrategy


def ensure_np_array(arr):
    if type(arr) != type(np.array([])):
        arr = np.array(arr).flatten()

    """
    Warning - can cause nested arrays for some algorithms
    e.g. OrderedRule learner. This code is probably to ensure output is column
        and not a row vector (which it does ensure if input has 1 dimension not multiple)
    """
    if len(arr.shape) == 1: #one not multi-dimensional
        arr = np.reshape(arr, (arr.shape[0], 1))
    return arr

class StackedExperimentRunner(object):
    __MIN_DOC_LENGTH__ = 3

    def get_ys(self, code, data, empty_ixs, label_mapper, xs):
        # remove ys for empty docs
        ys = [label_mapper(y) for i, y in enumerate(data.labels_for(code)) if i not in empty_ixs]
        ys = ensure_np_array(ys)
        assert len(ys) == len(xs)
        return ys

    def __extract_predictions__(self, codes, d_preds, td_x):
        preds = [[] for x in td_x]
        for code in codes:
            p = d_preds[code]
            for i in range(len(td_x)):
                preds[i].append(p[i])
        return preds

    def RunStacked(self, results_file, cv_folds = 10, min_word_count = 5,
                   stem = True, lemmatize = False, remove_stop_words = True, layers = 2):

        #SETTINGS
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        print "Results filename: " + results_file
        settings = Settings.Settings()

        results_dir = settings.results_directory + self.sub_dir() + "\\"

        fName = results_dir + results_file

        #TOKENIZE
        data = self.get_data(settings)
        tokenized_docs = WordTokenizer.tokenize(data.documents, min_word_count=min_word_count, stem=stem,
                                                lemmatize=lemmatize, remove_stop_words=remove_stop_words,
                                                spelling_correct=True, number_fn=NumberStrategy.collapse_num)

        empty_ixs = set([i for i, doc in enumerate(tokenized_docs) if len(doc) < StackedExperimentRunner.__MIN_DOC_LENGTH__])
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

        MIN_CODE_COUNT = 3

        codes = set(self.get_codes(data.sm_codes))
        label_mapper = self.label_mapper()

        # Stop logging now
        logging.disable(logging.INFO)

        xs = ensure_np_array(xs)
        edges = cross_validation_edges(len(xs), cv_folds)

        ys_by_code = {}
        positive_count_by_code = {}
        for code in codes.copy():
            ys = self.get_ys(code, data, empty_ixs, label_mapper, xs)
            ys_by_code[code] = ys

            positive_count = len([item for item in ys if item == 1])
            positive_count_by_code[code] = positive_count

            if positive_count < MIN_CODE_COUNT:
                codes.remove(code)

        dct_td_predictions_by_fold = {}
        dct_vd_predictions_by_fold = {}
        dct_actual_by_fold = {}

        for layer in range(layers):

            print("Layer: {0}".format(layer))
            vd_metrics_for_layer, td_metrics_for_layer = [], []

            vd_metrics_by_code = defaultdict(lambda: [])
            td_metrics_by_code = defaultdict(lambda: [])

            for fold in range(cv_folds):

                l, r = edges[fold]

                #Note these are numpy obj's and cannot be treated as lists
                td_x = np.concatenate((xs[:l], xs[r:]))
                vd_x = xs[l:r]

                predictions_from_previous_layer = None
                if layer > 0:
                    # Seed with an empty lists
                    lst_td_preds = self.__extract_predictions__(codes, dct_td_predictions_by_fold[fold], td_x)
                    td_x = np.concatenate((td_x, np.array(lst_td_preds)), 1)

                    lst_vd_preds = self.__extract_predictions__(codes, dct_vd_predictions_by_fold[fold], vd_x)
                    vd_x = np.concatenate((vd_x, np.array(lst_vd_preds)), 1)

                dct_td_predictions_per_code = {}
                dct_vd_predictions_per_code = {}
                dct_actual_per_code = {}

                dct_td_predictions_by_fold[fold] = dct_td_predictions_per_code
                dct_vd_predictions_by_fold[fold] = dct_vd_predictions_per_code
                dct_actual_by_fold[fold] = dct_actual_per_code

                class_value = self.get_class_value()

                for code in codes:

                    total_codes = positive_count_by_code[code]

                    ys = ys_by_code[code]
                    td_y = np.concatenate((ys[:l], ys[r:]))
                    vd_y = ys[l:r]

                    if min(td_y) == max(td_y):
                        val = td_y[0]
                        td_predictions = np.array([val for y in td_y])
                        vd_predictions = np.array([val for y in vd_y])
                    else:
                        create_classifier_func = self.create_classifier(code)
                        classify_func = self.classify()

                        classifier = create_classifier_func(td_x, td_y)
                        td_predictions = classify_func(classifier, td_x)
                        vd_predictions = classify_func(classifier, vd_x)

                    dct_td_predictions_per_code[code]  = td_predictions
                    dct_vd_predictions_per_code[code]  = vd_predictions
                    dct_actual_per_code[code]       = td_y

                    td_r, td_p, td_f1, td_a = Metrics.rpf1a(td_y, td_predictions, class_value=class_value)
                    vd_r, vd_p, vd_f1, vd_a = Metrics.rpf1a(vd_y, vd_predictions, class_value=class_value)

                    vd_metric, td_metric = self.rpfa(vd_r, vd_p, vd_f1, vd_a, total_codes), \
                                           self.rpfa(td_r, td_p, td_f1, td_a, total_codes)

                    vd_metrics_for_layer.append(vd_metric)
                    td_metrics_for_layer.append(td_metric)

                    vd_metrics_by_code[code].append(vd_metric)
                    td_metrics_by_code[code].append(td_metric)

                pass # End for code in codes

            pass #END for fold in folds

            for code in sorted(codes):
                positive_count = positive_count_by_code[code]
                vd_metric, td_metric = self.mean_rpfa(vd_metrics_by_code[code]), self.mean_rpfa(td_metrics_by_code[code])

                results = "Code: {0} Count: {1} VD[ {2} ]\tTD[ {3} ]\n".format(code.ljust(7), str(positive_count).rjust(4),
                                                                               vd_metric.to_str(), td_metric.to_str())
                print results,

            mean_vd_metrics, mean_td_metrics = self.mean_rpfa(vd_metrics_for_layer), self.mean_rpfa(td_metrics_for_layer)
            wt_mean_vd_metrics, wt_mean_td_metrics = self.weighted_mean_rpfa(vd_metrics_for_layer), self.weighted_mean_rpfa(
                td_metrics_for_layer)

            aggregate_results = "\n"
            aggregate_results += "VALIDATION DATA -\n"
            aggregate_results += "\tMEAN\n\t\t {0}\n".format(mean_vd_metrics.to_str(True))
            aggregate_results += "\tWEIGHTED MEAN\n\t\t {0}\n".format(wt_mean_vd_metrics.to_str(True))

            aggregate_results += "\n"
            aggregate_results += "TRAINING DATA -\n"
            aggregate_results += "\tMEAN\n\t\t {0}\n".format(mean_td_metrics.to_str(True))
            aggregate_results += "\tWEIGHTED MEAN\n\t\t {0}\n".format(wt_mean_td_metrics.to_str(True))

            print aggregate_results
            pass #End for layer in layers

        pass #End fold

        """ Dump results to file in case of crash """

        #DUMP TO FILE
        """
        print "Writing results to: " + fName
        handle = open(fName, mode="w+")
        handle.write(all_results)
        handle.close()
        """
        #return (mean_vd_metrics, wt_mean_vd_metrics)