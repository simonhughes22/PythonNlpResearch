import numpy as np

from Metrics import rpf1a
from Rpfa import rpfa

from gensim import matutils
from numpy import random
from collections import defaultdict
from IterableFP import compact
from os import listdir
from os.path import isfile, join
from Essay import Essay
from WindowSplitter import split_into_windows
from IdGenerator import IdGenerator

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.lda import LDA

def linux_style_path(fpath):
    return fpath.replace("\\", "/")

def analyse_essays(essays):
    code2txt = defaultdict(list)
    for essay in essays:
        for concept in essay.concepts:
            code2txt[concept.code].append(concept.txt)

    mean_lens = {}
    for code in sorted(code2txt.keys()):
        v = code2txt[code]
        v.sort()
        print code

        lens = []
        for txt in v:
            print "\t" + txt
            lngth = len(compact(txt.split(" ")))
            lens.append(lngth)
        mean_lens[code] = np.mean(lens)

def load_essays(root_folder):

    onlyfiles = [f for f in listdir(root_folder) if isfile(join(root_folder, f))]
    full_paths = map(lambda f: join(root_folder, f), onlyfiles)
    # Make linux style path
    full_paths = map(linux_style_path, full_paths)
    assert len(full_paths) == 105, \
        "Wrong number of files found: %d. Expected  - 105" % len(full_paths)
    print "%d files found" % len(full_paths)

    return map(Essay, full_paths)

""" Start Script """
WINDOW_SIZE = 5

SENTENCE_START  = "SENTENCE_START"
SENTENCE_END    = "SENTENCE_END"
MIN_SENTENCE_FREQ = 5
PCT_VALIDATION  = 0.2

essays = load_essays(root_folder = 'C:\Users\simon.hughes\Dropbox\PhD\Data\Coral Bleaching\Final XMLs\Final XMLs')

all_wds = set([SENTENCE_START, SENTENCE_END])
all_codes = set()

MID_IX = int(round(WINDOW_SIZE / 2.0) - 1)
windows = []
wd_sent_freq = defaultdict(int)

for essay in essays:
    for sentence in essay.tagged_sentences:

        wds = zip(*sentence)[0]
        for w in set(wds):
            wd_sent_freq[w] += 1

print "Creating Windows"
for essay in essays:
    for sentence in essay.tagged_sentences:

        for wd, tags in sentence:
            all_wds.add(wd)
            if len(tags) > 0:
                all_codes.update(tags)

        modified_sentence = [(w, tags) for (w, tags) in sentence if wd_sent_freq[w] >= MIN_SENTENCE_FREQ]
        if len(modified_sentence) == 0:
            continue

        # pad the start so features before the target word
        # also ensure that sentence is at least window size long
        for i in range(MID_IX):
            modified_sentence.insert(0, (SENTENCE_START,    set()))
            modified_sentence.append(   (SENTENCE_END,      set()))

        new_windows = split_into_windows(modified_sentence, window_size= WINDOW_SIZE)
        lens = map(len, new_windows)
        assert min(lens) == max(lens) == WINDOW_SIZE, "Windows are not all the correct size"
        windows.extend(new_windows)

idgen = IdGenerator()

def extract_tags(window):
    target_wd, tags = window[MID_IX]
    return tags

def extract_features(window):
    feats = {}

    for i, (wd, tags) in enumerate(window):
        feature_name = "WD:" + str(-MID_IX + i) + " " + wd
        feats[feature_name] = 1

    return feats

xs = []
ysByCode = defaultdict(list)

print "Extracting Features"
for window in windows:
    features = extract_features(window)
    tags = extract_tags(window)

    x = []
    for feat, val in features.items():
        id = idgen.get_id(feat)
        x.append( (id, val) )
    xs.append(x)

    for code in tags:
        ysByCode[code].append(1)
    for code in all_codes - tags:
        ysByCode[code].append(0)

assert len(xs) == len(ysByCode.values()[0])

""" Make numpy arrays """
num_features = idgen.max_id() + 1

""" Sparse to full """
xs = [matutils.sparse2full(x, num_features) for x in xs]
xs = np.asarray(xs)

print "Number of windows: ", len(xs)
print "Number of features:", num_features

for k, v in ysByCode.items():
    ysByCode[k] = np.asarray(v)

""" Compute TD \ VD split """
num_validation = int(0.2 * len(xs))
num_training = len(xs) - num_validation

printed_classifier = False

print "Training"
for code in sorted(all_codes):
    #print "Processing code:", code

    ys = ysByCode[code]
    num_codes = sum(ys)

    ixs = range(len(xs))
    random.shuffle(ixs)

    td_ixs, vd_ixs              = ixs[:num_training], ixs[num_training:]
    td_xs, vd_xs, td_ys, vd_ys  = xs[td_ixs], xs[vd_ixs], ys[td_ixs], ys[vd_ixs]

    #cls = LogisticRegression(penalty="l2", dual=True)
    #cls = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2)
    #cls = LinearSVC()
    """ Currently running """
    cls = GradientBoostingClassifier()
    #cls = RandomForestClassifier()
    #cls = LDA()

    if printed_classifier == False:
        print "Classifier:", str(cls)
        printed_classifier = True

    cls.fit(td_xs, td_ys)

    td_preds = cls.predict(td_xs)
    vd_preds = cls.predict(vd_xs)

    """ Training Data """
    r, p, f1, a = rpf1a(expected=td_ys, actual=td_preds, class_value=1)
    td_metric = rpfa(r, p, f1, a, num_codes)

    r, p, f1, a = rpf1a(expected=vd_ys, actual=vd_preds, class_value=1)
    vd_metric = rpfa(r, p, f1, a, num_codes)

    results = "Code: {0} Count: {1} VD[ {2} ]\tTD[ {3} ]\n".format(code.ljust(7), str(num_codes).rjust(4),
                                                                   vd_metric.to_str(), td_metric.to_str())
    print results,
""" thoughts:

    a lot of word overlap in the key phrases we are looking for - apriori?
    look for commonly occurring character n-grams (*bleach* *temp*, *n't*, etc)
    code 7 in particular
        - sentiment analysis on the window - words like die, murder, eject
        - lot's of variety in phrases for code 7, but many negative words
    use wordnet - equate words like die, death, murder
    have a negation tagger - detects phrases like "not", ...n't, don't, without - detect negative phrases
"""

