from collections import defaultdict
import numpy as np
import scipy

CAUSAL_REL   = "CRel"
RESULT_REL   = "RRel"
CAUSE_RESULT = "C->R"

def extract_ys_by_code(tags, all_codes, ysByCode):
    for code in all_codes:
        ysByCode[code].append(1 if code in tags else 0 )

    ysByCode[CAUSAL_REL].append(  1 if  "Causer" in tags and "explicit" in tags else 0)
    ysByCode[RESULT_REL].append(  1 if  "Result" in tags and "explicit" in tags else 0)
    ysByCode[CAUSE_RESULT].append(1 if ("Result" in tags and "explicit" in tags and "Causer" in tags) else 0)

def get_sent_feature_for_stacking(feat_tags, interaction_tags, essays, word_feats, ys_bytag, tag2Classifier, sparse=False):

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
    for essay_ix, essay in enumerate(essays):
        for sent_ix, taggged_sentence in enumerate(essay.sentences):
            tmp_xs = []

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

                tmp_xs.append(mx)
                tmp_xs.append(mn)
                # for val in mx:
                #tmp_xs.append(val)
                #tmp_xs.append(mx[-1])

                #for val in mn:
                #tmp_xs.append(val)
                #tmp_xs.append(mn[-1])

                yes_no = np.max(pred)
                tmp_xs.append(yes_no)

                if yes_no > 0.0:
                    un_pred_tags.add(tag)
                    # add 2 way feature combos

            #pairwise interactions
            for a in interaction_tags:
                for b in interaction_tags:
                    if b < a:
                        if a in un_pred_tags and b in un_pred_tags:
                            tmp_xs.append(1)
                        else:
                            tmp_xs.append(0)

            extract_ys_by_code(un_tags, feat_tags, ys_by_code)
            td_sent_feats.append(tmp_xs)

    for k in ys_by_code.keys():
        ys_by_code[k] = np.asarray(ys_by_code[k])

    assert len(td_sent_feats) == len(ys_by_code[ys_by_code.keys()[0]])
    if sparse:
        xs = scipy.sparse.csr_matrix(td_sent_feats)
    else:
        xs = np.asarray(td_sent_feats)
    return xs, ys_by_code