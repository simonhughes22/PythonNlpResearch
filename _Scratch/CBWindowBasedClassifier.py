
""" Imports """
from collections import defaultdict

import numpy as np
from gensim import matutils
from numpy import random

from Metrics import rpf1a
from Rpfa import rpfa, weighted_mean_rpfa, mean_rpfa
from BrattEssay import load_bratt_essays
from WindowSplitter import split_into_windows

from IdGenerator import IdGenerator
from IterableFP import flatten

from nltk import PorterStemmer
from stanford_parser import parser

""" TODO
    Try dependency parse features from this python dependency parser: https://github.com/syllog1sm/redshift
"""
None
""" Settings """
""" Start Script """
WINDOW_SIZE = 7 #7 is best
MID_IX = int(round(WINDOW_SIZE / 2.0) - 1)

MIN_SENTENCE_FREQ = 2
PCT_VALIDATION  = 0.2
MIN_FEAT_FREQ = 5     #15 best so far
PCT_VALIDATION = 0.25

SENTENCE_START = "<START>"
SENTENCE_END   = "<END>"
STEM = True

""" Load Essays """
essays = load_bratt_essays("/Users/simon.hughes/Dropbox/Phd/Data/CoralBleaching/BrattData/Merged/")

all_codes = set()
all_words = []

CAUSAL_REL = "CRel"
RESULT_REL = "RRel"
CAUSE_RESULT = "C->R"

cr_codes = [CAUSAL_REL, RESULT_REL, CAUSE_RESULT]

for essay in essays:
    for sentence in essay.tagged_sentences:
        for w, tags in sentence:
            all_words.append(w)
            all_codes.update(tags)

# Correct miss-spellings
from SpellingCorrector import SpellingCorrector

corrector = SpellingCorrector(all_words)
corrections = defaultdict(int)
for essay in essays:
    for i, sentence in enumerate(essay.tagged_sentences):
        for j, (w, tags) in enumerate(sentence):
            # common error is ..n't and ..nt
            if w.endswith("n't") or w.endswith("n'"):
                cw = w[:-3] + "nt"
            elif w.endswith("'s"):
                cw = w[:-2]
            elif w == "&":
                cw = "and"
            else:
                cw = corrector.correct(w)
            if cw != w:
                corrections[(w,cw)] += 1
                sentence[j] = (cw, tags)

wd_sent_freq = defaultdict(int)
for essay in essays:
    for sentence in essay.tagged_sentences:
        wds, tag_list = zip(*sentence)
        unique_wds = set(wds)
        for w in unique_wds:
            wd_sent_freq[w] += 1
print "Done"

""" Creating Windows """
def filter2min_word_freq(sentence):
    return filter(lambda (w, tags4word): wd_sent_freq[w] >= MIN_SENTENCE_FREQ, sentence)

VALID_CHARS = {".", "?", "!", "=", "/", ":", ";", "&", "+",  "-", "=",  "%", "'", ",", "\\", "(", ")", "\""}
""" Remove bad chars (see above - e.g. '\x93') """
removed = set()
def valid_wd(wd):
    wd = wd.strip()
    if len(wd) != 1:
        return True
    if wd in removed:
        return False
    if wd.isalpha() or wd.isdigit() or wd in VALID_CHARS:
        return True
    removed.add(wd)
    return False

def filterout_punctuation(sentence):
    return filter(lambda (w, tags4word): valid_wd(w), sentence)

def bookend(sentence):
    for i in range(MID_IX):
        modified_sentence.insert(0, (SENTENCE_START,    set()))
        modified_sentence.append(   (SENTENCE_END,      set()))

def assert_windows_correct(windows):
    lens = map(len, windows)
    assert min(lens) == max(lens) == WINDOW_SIZE, \
            "Windows are not all the correct size"

ix2windows = {}
ix2sents = {}
ix2sentTags = {}

sentences = []
tokenized_sentences = []

i = 0
for essay in essays:
    for sentence in essay.tagged_sentences:

        modified_sentence = filter2min_word_freq(sentence)
        modified_sentence = filterout_punctuation(modified_sentence)
        if len(modified_sentence) == 0:
            continue

        bookend(modified_sentence)
        new_windows = split_into_windows(modified_sentence, window_size= WINDOW_SIZE)
        assert_windows_correct(new_windows)

        # tagged words
        sentences.append(sentence)
        # words only
        wds, tags = zip(*sentence)
        tokenized_sentences.append(wds)
        ix2sentTags[i] = set(flatten(tags))

        ix2windows[i] = new_windows
        ix2sents[i] = modified_sentence
        i += 1

""" Assert tags set correctly """
print "Windows loaded correctly!\n"

print "\n".join(sorted(removed))

""" Extract Features """
from WindowFeatures import extract_positional_word_features, extract_word_features
from NgramGenerator import compute_ngrams

def extract_positional_bigram_features(window, mid_ix, feature_val = 1):
    bi_grams = compute_ngrams(window, max_len = 2, min_len = 2)
    d = {}
    for i, bi_gram in enumerate(bi_grams):
        d["BI" + ":" + str(-mid_ix + i) + " " + bi_gram[0] + " | " + bi_gram[1]] = feature_val
    return d

""" TODO:
        Extract features for numbers
        Extract features for years
        Extract features that are temperatures (look for degree\degrees in subsequent words, along with C or F)
"""
idgen = IdGenerator()
stemmer = PorterStemmer()

def extract_features(words):

    if STEM:
        words = [stemmer.stem(w) for w in words]

    #Extract features for words
    features = {}
    pos_features = extract_positional_word_features(words, MID_IX, feature_val=1)
    word_features  = extract_word_features(words, feature_val=1)
    pos_bi_grams = extract_positional_bigram_features(words, MID_IX, feature_val = 1)

    features.update(pos_features)
    features.update(word_features)
    features.update(pos_bi_grams)
    return features.items()

def extract_ys_by_code(tags, ysByCode):
    for code in all_codes:
        ysByCode[code].append(1 if code in tags else 0 )

    ysByCode[CAUSAL_REL].append(  1 if  "Causer" in tags and "explicit" in tags else 0)
    ysByCode[RESULT_REL].append(  1 if  "Result" in tags and "explicit" in tags else 0)
    ysByCode[CAUSE_RESULT].append(1 if ("Result" in tags and "explicit" in tags and "Causer" in tags) else 0)

ix2ys = {}
ix2feats = {}
feat_counts = defaultdict(int)
def tally_features(feats):
    for k,v in feats:
        feat_counts[k] += 1

for i, windows in ix2windows.items():
    feats = []
    ysByCode = defaultdict(list)

    ix2feats[i] = feats
    ix2ys[i] = ysByCode
    for window in windows:
        # Get the words minus tags
        words, tags = zip(*window)
        feat = extract_features(words)
        tally_features(feat)
        feats.append(feat)

        #Tags for middle word (target)
        tags4word = tags[MID_IX]
        extract_ys_by_code(tags4word, ysByCode)
    assert len(windows) == len(feats)
    assert all(map(lambda (k,v): len(v) == len(feats), ysByCode.items()))

""" Convert sparse dictionary features to sparse arrays """
ix2xs = {}
for i, feature_lists in ix2feats.items():
    xs = []
    ix2xs[i] = xs
    for feats in feature_lists:
        x = [(idgen.get_id(f),v)
             for f,v in feats
             if feat_counts[f] >= MIN_FEAT_FREQ or f.startswith("WD:0" )] #above min freq or is word
        xs.append(x)

num_features = idgen.max_id() + 1
print "Number of features:", num_features

""" Convert to dense numpy arrays """
for i in ix2xs.keys():
    xs = ix2xs[i]
    xs = np.array([matutils.sparse2full(x, num_features) for x in xs])
    ix2xs[i] = xs
from DictionaryHelper import *

def count_above(ft_counts, threshold):
    above = [ v for k,v in ft_counts.items() if v >= threshold]
    return (len(above), len(ft_counts))

cnt_above, cnt_all = count_above(feat_counts, MIN_FEAT_FREQ)

print "Counts"
print "all:     ", cnt_all
print "above:   ", cnt_above
print "% above: ", str(100.0 * cnt_above / float(cnt_all))+ "%"

""" Extract ys for sentence (including causal codes) """
ix2ys_sent = {}
for i, tags in ix2sentTags.items():
    ix2ys_sent[i] = defaultdict(list)
    extract_ys_by_code(tags, ix2ys_sent[i])
""" Create Data Using Previous Classifier """

def to_sentence_level_predictions(ix2xs, code2cls):
    ix2newxs = {}

    for i, xs in ix2xs.items():
        tmp_xs = []
        tmp_ys = []
        tmp_ys_by_code = defaultdict(list)

        un_codes = set()
        un_pred_codes = set()
        for code in all_codes:
            cls = code2cls[code]
            pred = cls.decision_function(xs)
            # add min and max values
            mx = max(pred)
            mn = min(pred)
            yes_no = max(cls.predict(xs))

            tmp_xs.append(mx)
            tmp_xs.append(mn)
            tmp_xs.append(yes_no)

            if yes_no > 0.0:
                un_pred_codes.add(code)

        #add 2 way feature combos
        for a in all_codes:
            for b in all_codes:
                if b < a:
                    if a in un_pred_codes and b in un_pred_codes:
                        tmp_xs.append(1)
                    else:
                        tmp_xs.append(0)

        ix2newxs[i] = np.array([tmp_xs])
    return ix2newxs
""" Get sentence level classification performance """
def test_for_code(code, ixs, ixToXs, ixToYs, codeToClassifier):
    cls = codeToClassifier[code]

    act_ys  = []
    pred_ys = []
    for ix in ixs:
        xs = ixToXs[ix]
        ysByCode = ixToYs[ix]

        ys = np.asarray(ysByCode[code])
        #ys = map(map_y, ys)
        pred = cls.predict(xs)

        # Flatten predictions to sentence level by taking the max values
        # over all windows
        act_ys.append(max(ys))
        pred_ys.append(max(pred))

    num_codes = len([y for y in act_ys if y == 1])
    r,p,f1,a = rpf1a(act_ys, pred_ys)
    return rpfa(r,p,f1,a,num_codes)

def test(codes, ixs, ixToXs, ixToYs, codeToClassifier):
    td_metrics = []
    for c in codes:
        cls = codeToClassifier[c]
        td_metrics.append(test_for_code(c, ixs, ixToXs, ixToYs, codeToClassifier))
    td_wt_mn_prfa = weighted_mean_rpfa(td_metrics)
    print type(cls), td_wt_mn_prfa
    return td_wt_mn_prfa
from CrossValidation import cross_validation

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier

def extract_xs_ys(ixs, ixTOxs, ixTOys, codes):
    xs = []
    ysByCode = defaultdict(list)
    for i in ixs:
        xs_tmp = ixTOxs[i]
        xs.extend(xs_tmp)
        ysByCode_tmp = ixTOys[i]
        for code in codes:
            ysByCode[code].extend(ysByCode_tmp[code])
    return (np.array(xs), ysByCode)

def train(codes, xs, yByCode, fn_create_cls):
    code2classifier = {}
    for code in codes:
        print "Training for :", code
        cls = fn_create_cls()
        code2classifier[code] = cls
        ys = np.asarray(yByCode[code])
        #ys = map(map_y, ys)
        cls.fit(xs, ys)
    return code2classifier

fn_classifier = LinearSVC
SPLITS = 2
causal_codes = cr_codes + ["explicit"]

ixs = range(len(sentences))
folds = cross_validation(ixs, SPLITS)
td_metrics = []
vd_metrics = []

for num, (ix_train, ix_valid) in enumerate(folds):
    print "Fold:", num + 1

    # Train sequential classifier
    xs_t, yByCode_t = extract_xs_ys(ix_train, ix2xs, ix2ys, all_codes)
    code2cls = train(all_codes, xs_t, yByCode_t, fn_classifier)

    print "Training Sentence Classifier"
    # Extract new data points and target classes
    ix2xs_sent = to_sentence_level_predictions(ix2xs, code2cls)
    newxs_t, newyByCode_t = extract_xs_ys(ix_train, ix2xs_sent, ix2ys_sent, all_codes + causal_codes)

    new_code2cls = train(causal_codes, newxs_t, newyByCode_t, fn_classifier)
    # Evaluate
    td_metrics.append(test(causal_codes, ix_train, ix2xs_sent, ix2ys_sent, new_code2cls))
    print "Test performance      ", td_metrics[-1]
    vd_metrics.append(test(causal_codes, ix_valid, ix2xs_sent, ix2ys_sent, new_code2cls))
    print "Validation performance", vd_metrics[-1]

print "\NFinished"
print "MEAN Test Performance      ", mean_rpfa(td_metrics)
print "MEAN Validation Performance", mean_rpfa(vd_metrics)
l = []
code = "CRel"
for i in ix2ys_sent.keys()[:400]:
    l.append(ix2ys_sent[i][code][0])
print min(l), max(l), len([1 for i in l if i > 0.0]), len(l)
