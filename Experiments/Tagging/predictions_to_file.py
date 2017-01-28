from collections import defaultdict

__author__ = 'simon.hughes'

from traceback import format_exc
import numpy as np

def predictions_to_file(file, ys_by_code, predictions_by_code, essays, codes = None, output_confidence=False):

    def sort_key(code):
        if code[0].isdigit():
            return (-1, len(code), code)
        return (1, len(code), code)

    if codes is None:
        codes = sorted(set(list(ys_by_code.keys()) + list(predictions_by_code.keys())))
    else:
        codes = sorted(set(codes))
    ix = 0
    for essay_ix, essay in enumerate(essays):
        for sent_ix, tagged_sentence in enumerate(essay.sentences):
            predictions = set()
            actual = set()
            for code in codes:
                try:
                    if code in ys_by_code:
                        y_val = ys_by_code[code][ix]
                        if y_val > 0:
                            actual.add(code)
                        # some codes are not yet predicted
                    if code in predictions_by_code:
                        pred_y_val = predictions_by_code[code][ix]
                        if output_confidence:
                            predictions.add("%s@%f" % (code, pred_y_val))
                        elif pred_y_val > 0:
                            predictions.add(code)
                except:
                    print format_exc()
                    raise Exception("Error processing code %s" % code)

            words = map(lambda ft: ft.word, tagged_sentence)
            s_words = " ".join(words)
            file.write("|".join([essay.name, str(sent_ix + 1), s_words, ",".join(sorted(actual, key=sort_key)), ",".join(sorted(predictions, key=sort_key))]))
            file.write("\n")
            ix += 1
    pass

def word_predictions_to_file(file, essays, word_feats, ys_bytag, tag2Classifier):

    # dicts, key = tag, to a 1D array of word-level predictions
    real_num_predictions_bytag = dict()
    for tag in tag2Classifier.keys():

        cls = tag2Classifier[tag]
        if hasattr(cls, "decision_function"):
            real_num_predictions = cls.decision_function(word_feats)
        else:
            real_num_predictions = cls.predict_proba(word_feats)
        real_num_predictions_bytag[tag] = real_num_predictions

    stags = sorted(ys_bytag.keys())

    # features for the sentence level predictions
    ix = 0
    for essay_ix, essay in enumerate(essays):

        for sent_ix, taggged_sentence in enumerate(essay.sentences):
            file.write(essay.name + "|")
            file.write(str(sent_ix + 1) + "|")

            # ixs into the tagged words
            ixs = range(ix, ix + len(taggged_sentence))
            ix += len(taggged_sentence)

            words = map(lambda ft: ft.word, taggged_sentence)
            for wix, word in enumerate(words):
                if wix > 0:
                    file.write(" ")
                word_ix = ixs[wix]
                file.write(word + "{")
                preds = []
                for tag in stags:
                    real_pred = real_num_predictions_bytag[tag][word_ix]
                    preds.append("%s:%f" % (tag, round(real_pred, 4)))
                file.write(",".join(preds))
                file.write("}")
            file.write("\n")
