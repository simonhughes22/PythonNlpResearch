# coding=utf-8
from collections import defaultdict
from IterableFP import flatten

INSIDE = "I"
OUTSIDE = "O"

# Computes the frequencies of each essay tag
def tally_code_frequencies(tagged_essays):
    freq = defaultdict(int)
    all_codes = set()
    for essay in tagged_essays:
        for i, sentence in enumerate(essay.sentences):
            words, tags = zip(*sentence)
            utags = set(flatten(tags))
            all_codes.update(utags)
            for t in utags:
                freq[t] += 1
    return freq

# Takes a list of essay objects and a list of target codes,
# and returns a dictionary of code to tagged sentences, where a
# tagged sentence is a list of tuples of (word, (INSIDE\OUTSIDE)
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

# Takes a list of essay objects and a list of target codes,
# and returns a list tagged sentences, where a tagged sentence
# is a list of tuples of (word, TAG) where tag is the most common
# associated TAG (only use for training and NOT for evaluation)
def to_most_common_code_tagged_sentences(essays, codes, code_freq):
    codes = set(codes)
    tagged = []
    for essay in essays:
        for i, sentence in enumerate(essay.sentences):
            sent = []
            for wd, tags in sentence:
                # filter to target codes only
                tags = codes.intersection(tags)
                if len(tags) > 0:
                    most_common = max(tags, key = lambda tag: code_freq[tag])
                    sent.append((wd, most_common))
                else:
                    sent.append((wd, OUTSIDE))
            tagged.append(sent)
    return tagged

# Takes a list of essay objects and a list of target codes,
# and returns a list tagged sentences, where a tagged sentence
# is a list of tuples of (word, POWERSET) where POWERSET is a
# conjunction of unique codes (where there are multiple tags for a word)
def to_label_powerset_tagged_sentences(essays, codes):
    codes = set(codes)
    tagged = []
    for essay in essays:
        for i, sentence in enumerate(essay.sentences):
            sent = []
            for wd, tags in sentence:
                # filter to target codes only
                isect_tags = ",".join(sorted(codes.intersection(tags)))
                if len(isect_tags) > 0:
                    # append as powerset label
                    sent.append((wd, isect_tags))
                else:
                    sent.append((wd, OUTSIDE))
            tagged.append(sent)
    return tagged

# Takes a set of tagged sentences (lists of tuples) and returns a list of sentences (list of tokens)
def to_sentences(tagged_sentences):
    sents = []
    for sentence in tagged_sentences:
        words, tags = zip(*sentence)
        sents.append(words)
    return sents

# Takes a set of INSIDE \ OUTSIDE predictions (nested lists of tuples)
# and converts to a singel flattened list of 1's, and 0's
def to_flattened_binary_tags(tagged_sentences):
    tags = []
    for sentence in tagged_sentences:
        words, lbls = zip(*sentence)
        tags.extend((1 if t == INSIDE else 0 for t in lbls))
    return tags

# Takes a set of tagged sentences, where the tags can be a set, a string, or a comma delimited string
# and pivots into a dictionary of binary labels (1's and 0's), one entry per target code
def to_flattened_binary_tags_by_code(tagged_sentences, codes):
    code2sents = defaultdict(list)
    for sentence in tagged_sentences:
        words, lbls = zip(*sentence)
        # for each word's tag (expects a single tag)
        for t in lbls:
            if type(t) != set:
                t = set(t.split(","))
            for code in codes:
                code2sents[code].append(1 if code in t else 0)
    return code2sents