from nltk.tag.crf import MalletCRF
import numpy as np

def feat_det(tokens, index):
    d = dict([(t, 1) for t in tokens])
    return d

corpus = [
    [("a", "1"), ("b", "1"), ("c", "2"), ("d", "2"), ("e", "2")],
    [("b", "1"), ("a", "1"), ("d", "2")],
]
test_sentences = [
    ["a", "d", "c"],
    ["a", "b", "e"],
    ["a", "d", "e"]
]

corpus = np.asarray(corpus)
test_sentences = np.asarray(test_sentences)

tagger = MalletCRF.train(feature_detector= feat_det,
                         corpus=corpus)
tags = tagger.batch_tag(test_sentences)
print "Tags:"
print "\n".join(map(str,tags))

#tagger.tag(["a", "d", "c"])
