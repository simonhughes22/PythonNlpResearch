from nltk.tag.crf import MalletCRF

def feat_det(tokens, index):
    return dict([(t, 1) for t in tokens])

tagger = MalletCRF.train(feature_detector= feat_det,
                         corpus=[ [("a","1"),("b", "1"), ("c", "2"), ("d", "2")] ])