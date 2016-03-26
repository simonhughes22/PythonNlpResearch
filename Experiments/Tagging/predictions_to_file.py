__author__ = 'simon.hughes'

from traceback import format_exc

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