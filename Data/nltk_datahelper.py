# coding=utf-8
from collections import defaultdict

INSIDE = "I"
OUTSIDE = "O"

def to_tagged_sentences_by_code(essays, codes, projection = lambda x:x):
    code2sents = defaultdict(list)
    for essay in essays:
        for i, sentence in enumerate(essay.sentences):
            for code in codes:
                sent = []
                for wd, tags in sentence:
                    wd = projection(wd)
                    if code in tags:
                        sent.append((wd, INSIDE))
                    else:
                        sent.append((wd, OUTSIDE))
                code2sents[code].append(sent)
    return code2sents


def to_sentences(tagged_sentences):
    sents = []
    for sentence in tagged_sentences:
        words, tags = zip(*sentence)
        sents.append(words)
    return sents


def to_flattened_binary_tags(tagged_sentences):
    tags = []
    for sentence in tagged_sentences:
        words, lbls = zip(*sentence)
        tags.extend((1 if t == INSIDE else 0 for t in lbls))
    return tags