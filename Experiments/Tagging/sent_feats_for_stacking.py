from collections import defaultdict
import numpy as np
import scipy

CAUSAL_REL   = "_CRel"
RESULT_REL   = "_RRel"
CAUSE_RESULT = "_C->R"

def extract_ys_by_code(tags, all_codes, ysByCode):
    for code in all_codes:
        ysByCode[code].append(1 if code in tags else 0 )

    ysByCode[CAUSAL_REL].append(  1 if  "Causer" in tags and "explicit" in tags else 0)
    ysByCode[RESULT_REL].append(  1 if  "Result" in tags and "explicit" in tags else 0)
    ysByCode[CAUSE_RESULT].append(1 if ("Result" in tags and "explicit" in tags and "Causer" in tags) else 0)

def num_different_tags(pred):
    wds_since_prev = 9999
    new_tag = False
    cnt = 0
    for has_tag in pred:
        if has_tag > 0:
            if wds_since_prev >= 2 and not new_tag:
                cnt += 1
                new_tag = True
            wds_since_prev = 0
        else:
            wds_since_prev += 1
            new_tag = False
    return cnt

def get_sent_feature_for_stacking(feat_tags, interaction_tags, essays, word_feats, ys_bytag, tag2Classifier, sparse=False, look_back=0):

    real_num_predictions_bytag = dict()
    predictions_bytag = dict()
    for tag in feat_tags:

        cls = tag2Classifier[tag]
        if hasattr(cls, "decision_function"):
            real_num_predictions = cls.decision_function(word_feats)
        else:
            real_num_predictions = cls.predict_proba(word_feats)
        predictions = cls.predict(word_feats)
        real_num_predictions_bytag[tag] = real_num_predictions
        predictions_bytag[tag] = predictions
        real_num_predictions_bytag[tag] = real_num_predictions

    # features for the sentence level predictions
    td_sent_feats = []
    ys_by_code = defaultdict(list)
    ix = 0
    lst_look_back = range(look_back)
    for essay_ix, essay in enumerate(essays):

        tmp_essays_xs = []
        for sent_ix, taggged_sentence in enumerate(essay.sentences):
            tmp_sentence_xs = []

            # unique
            un_tags = set()
            un_pred_tags = set()

            ixs = range(ix, ix + len(taggged_sentence))
            ix += len(taggged_sentence)
            for tag in feat_tags:
                ys = ys_bytag[tag][ixs]
                if np.max(ys, axis=0) > 0:
                    un_tags.add(tag)

                real_pred = real_num_predictions_bytag[tag][ixs]
                pred = predictions_bytag[tag][ixs]

                assert ys.shape[0] == real_pred.shape[0] == pred.shape[0]
                mx = np.max(real_pred, axis=0)
                mn = np.min(real_pred, axis=0)

                tmp_sentence_xs.append(mx)
                tmp_sentence_xs.append(mn)

                yes_no = np.max(pred)
                tmp_sentence_xs.append(yes_no)

                if yes_no > 0.0:
                    un_pred_tags.add(tag)
                    # for adding 2-way feature combos

            #pairwise interactions (2 way interactions of predicted tags)
            for a in interaction_tags:
                for b in interaction_tags:
                    if b < a:
                        if a in un_pred_tags and b in un_pred_tags:
                            tmp_sentence_xs.append(1)
                        else:
                            tmp_sentence_xs.append(0)

            extract_ys_by_code(un_tags, feat_tags, ys_by_code)
            tmp_essays_xs.append(tmp_sentence_xs)
            # end sentence processing

        feats_per_sentence = len(tmp_essays_xs[0])
        blank = [0] * feats_per_sentence

        for i, sent_feats in enumerate(tmp_essays_xs):
            concat_feats = list(sent_feats)
            offset = -1
            for j in lst_look_back:
                ix = i + offset
                if ix < 0:
                    to_add = blank
                else:
                    to_add = tmp_essays_xs[ix]
                concat_feats.extend(to_add)
                offset -= 1
            td_sent_feats.append(concat_feats)

    for k in ys_by_code.keys():
        ys_by_code[k] = np.asarray(ys_by_code[k])

    assert len(td_sent_feats) == len(ys_by_code[ys_by_code.keys()[0]])
    if sparse:
        xs = scipy.sparse.csr_matrix(td_sent_feats)
    else:
        xs = np.asarray(td_sent_feats)
    return xs, ys_by_code

if __name__ == "__main__":


    def test(arr):
        print num_different_tags(arr), "->","".join(map(str,arr))

    test([0,0,1,0,1,1,0])
    test([0,0,1,0,0,1,1,0,0,0,1,1,1])
    test([1,0,0,1])
    test([1,0,0,1,0,0,1,0,0,1])
    test([1,0,1,0,1,0,1])