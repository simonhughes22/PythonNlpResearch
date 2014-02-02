__author__ = 'simon.hughes'

from itertools import izip
from Entropy import entropy
from collections import defaultdict
import numpy as np
import re

class Rule(object):
    def __init__(self, accuracy, coverage, words, class_matched):
        self.accuracy = accuracy
        self.coverage = coverage
        self.words = words
        self.class_matched = class_matched

    def matches(self, sent_wds):

        tmp_wds = self.words[:]
        for sw in sent_wds:
            for next_wd in tmp_wds:
                if sw == next_wd:
                    tmp_wds.pop(0)
                    break

            if len(tmp_wds) == 0:
                return True
        return False

    def __repr__(self):
        return "[{0}] -> {1}\t. Accuracy: {2}, Coverage: {3}\n".format(self.class_matched, self.words, self.accuracy, self.coverage)

class OrderedRuleLearner(object):

    def __init__(self):
        """ default to current """
        self.NEGATIVE_VAL = 0
        self.positive_val = 1
        self.class_values = [self.NEGATIVE_VAL, self.positive_val]
        self.rules = []

    """ private """
    def __create_rule__(self, words, all_examples, all_labels):
        coverage = 0.0
        correct = 0.0

        tmp_rule = Rule(-1, -1, words, -9999)
        pos_cnt = 0
        neg_cnt = 0
        for (wds, sentence), label in izip(all_examples, all_labels):
            if tmp_rule.matches(wds):
                if label == self.positive_val:
                    coverage += 1.0
                    correct += 1.0
                    pos_cnt +=1
                else:
                    neg_cnt +=1
            else:
                if label == self.NEGATIVE_VAL:
                    correct += 1.0

        cnt = len(all_examples)
        accuracy = correct / cnt
        matched_class = self.positive_val if pos_cnt > neg_cnt else self.NEGATIVE_VAL
        return Rule(accuracy, coverage / cnt, words, matched_class)

    def __eval__(self, pattern_words, examples, labels):

        matches = []
        tally = defaultdict(int)
        r = Rule(-1,-1, pattern_words, -99999)

        for (sent_wds, sentence), label in izip(examples, labels):

            if r.matches(sent_wds):
                matches.append(label)
                tally[label] += 1

        # Do some smoothing. This handles both the zero matches case and the
        # 0 entropy case (either no matches - bad, or all are the same - very good)
        for v in self.class_values:
            matches.append(v)
            tally[v] += 1
        lbl, lbl_cnt = sorted(tally.items(), key=lambda (k, v): -v)[0]
        return lbl_cnt / entropy(matches)
        #return 1.0 / entropy(matches)

    def __get_best_rules__(self, new_rules, examples, labels):
        best = []
        best_val = -1
        for r in new_rules:
            val = self.__eval__(r, examples, labels)
            if val > best_val:
                best = [r]
                best_val = val
            elif val == best_val:
                best.append(r)
        return (best, best_val)

    def __get_candidate_terms__(self, examples, min_freq=0.05):
        tally = defaultdict(int)
        for (words, sentence) in examples:
            for word in set(words):
                tally[word] += 1

        min_cnt = len(examples) * min_freq * 1.0
        return set((w for (w,cnt) in tally.items() if cnt >= min_cnt))

    def __get_fore_and_after_map__(self, positive_examples):
        b4_map = defaultdict(set)
        after_map = defaultdict(set)

        for ex_wds, _ in positive_examples:
            for i, wd in enumerate(ex_wds):
                # Can only have words b4 if not the first word
                b4_map[wd].update(ex_wds[:i])
                after_map[wd].update(ex_wds[i + 1:])
        return (b4_map, after_map)

    def __generate_new_rules__(self, seed, b4_map, after_map):

        new_rules = []
        for insertion_point in range(len(seed) + 1):

            if insertion_point == 0:
                candidates = b4_map[seed[0]]
            elif insertion_point == len(seed):
                candidates = after_map[seed[-1]]
            else:
                wd = seed[insertion_point]
                candidates = b4_map[wd].intersection(after_map[wd])

            for c in candidates:
                new_rule = seed[:insertion_point] + [c] + seed[insertion_point:]
                new_rules.append(new_rule)
        return new_rules

    def __get_next_rule__(self, uncovered_examples, uncovered_labels, all_examples, all_labels):
        positive_examples = self.__get_positive_examples__(uncovered_examples, uncovered_labels)
        if len(positive_examples) == 0:
            return None
        b4_map, after_map = self.__get_fore_and_after_map__(positive_examples)

        best_val = float('-inf')
        best_pattern = []

        while True:
            if len(best_pattern) == 0:
                new_rules = [[term] for term in self.__get_candidate_terms__(positive_examples)]
            else:
                new_rules = self.__generate_new_rules__(best_pattern, b4_map, after_map)
                if len(new_rules) == 0:
                    break

            best_rules, val = self.__get_best_rules__(new_rules, uncovered_examples, uncovered_labels)

            if len(best_rules) == 1:
                new_best_rule = best_rules[0]
            else:
                # multiple best rules. Pick the best on the entire dataset
                best_overall_rules, _ = self.__get_best_rules__(best_rules, all_examples, all_labels)
                new_best_rule = best_overall_rules[0]

            # Stop when no improvement
            if val <= best_val:
                break
            best_pattern, best_val = new_best_rule, val
        return self.__create_rule__(best_pattern, all_examples, all_labels)


    def __get_positive_examples__(self, examples, labels):
        return [ex for (ex, lbl) in izip(examples, labels) if lbl == self.positive_val]

    def __get_positive_val__(self, positives):
        if len(positives) > 0:
            return max(positives)
        """ Else default to current """
        return self.positive_val

    def __get_positive_ys__(self, ys):
        return np.array([y for y in ys if self.__positive_test__(y)]).flatten()

    def __get_unmatched_sentences__(self, uncovered_examples, uncovered_labels):
        return list(izip(* [(  (words, sentence), label   )
                for (words, sentence), label in izip(uncovered_examples, uncovered_labels)
                if not self.__matches_any_rule__(words)]))

    def __matches_any_rule__(self, sent_wds):
        return any(r for r in self.rules if r.matches(sent_wds))

    def __positive_test__(self, yval):
        return yval != self.NEGATIVE_VAL

    """ end private """

    def fit(self, xs, ys):
        """ [xs] : a list of sentences
            [ys] : labels - 0 (negative) non-zero (positive)
        """
        positive_labels = self.__get_positive_ys__(ys)
        self.positive_val = self.__get_positive_val__(positive_labels)
        self.class_values = [self.NEGATIVE_VAL, self.positive_val]

        all_examples = [(words, " ".join(words)) for words in xs[:]]
        uncovered_examples = all_examples[:]
        uncovered_labels = ys[:]

        """ A list of tuples (accuracy, coverage, pattern) """
        self.rules = []
        while True:
            next_rule = self.__get_next_rule__(uncovered_examples, uncovered_labels, all_examples, ys)
            if next_rule == None:
                break
            self.rules.append(next_rule)
            tpl = self.__get_unmatched_sentences__(uncovered_examples, uncovered_labels)
            if len(tpl) == 0:
                break
            uncovered_examples, uncovered_labels = tpl

    def predict(self, xs):
        return [self.positive_val if self.__matches_any_rule__(words) else self.NEGATIVE_VAL
                for words in xs]
        """ TODO BETTER matching strategy """
    #End class
    pass

if __name__ == "__main__":

    from Metrics import rpf1a
    import datetime

    def timeit(fn, label):
        begin = datetime.datetime.now()
        print label + " at ", begin
        rslt = fn()
        assert rslt != None
        stopped = datetime.datetime.now()
        print label + " took " + str((stopped - begin).total_seconds()) + " seconds"
        return rslt


    """ Test Code """

    #positive = chasing phrases
    td_xs = [
        ["the", "cat", "galloped", "towards", "the", "rabbit"],
        ["the", "cat", "galloped", "for", "the", "rabbit"],
        ["the", "cat", "raced", "to", "the", "rabbit"],
        ["the", "cat", "pursued", "the", "rabbit"],
        ["a", "dog", "sped", "towards", "an", "oxen"],
        ["a", "dog", "sped", "in", "the", "direction", "of", "an", "oxen"],
        ["the", "cat", "was", "chased", "by", "the", "dog"],
        ["the", "mouse", "was", "chased", "by", "the", "owl"],
        ["the", "man", "ran", "after", "the", "fox"],
        ["the", "cat", "ran", "for", "the", "rabbit"],
        ["a", "dog", "chased", "an", "oxen"],
        ["the", "cat", "was", "chased", "by", "the", "dog"],
        ["the", "mouse", "was", "chased", "by", "the", "owl"],
        ["the", "man", "ran", "after", "the", "fox"],

        ["the", "man", "ate", "the", "partridge"],
        ["the", "man", "shot", "the", "fox"],
        ["a", "man", "was", "sunbathing"],
        ["a", "dog", "was", "chewing", "grass"],
        ["the", "weather", "was", "warm"],
        ["the", "man", "ate", "the", "partridge"],
        ["the", "man", "shot", "the", "fox"],
        ["a", "man", "was", "sunbathing"],
        ["a", "dog", "was", "chewing", "grass"],
        ["the", "weather", "was", "warm"],
    ]
    td_ys = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,

        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    vd_xs = [ #positive = chasing
        ["the", "man", "chased", "the", "cat"],
        ["the", "bull", "chased", "the", "man"],
        ["the", "man", "ran", "after", "the", "fox"],
        ["the", "fox", "chased", "the", "rabbit"],

        ["the", "fox", "ate", "some", "meat"],
        ["a", "fox", "was", "grooming", "itself"],
        ["a", "cat", "was", "chewing", "grass"],
        ["the", "dog", "ate", "the", "rabbit"],
    ]
    vd_ys = [
        1,
        1,
        1,
        1,

        0,
        0,
        0,
        0,
    ]

    assert len(td_xs) == len(td_ys), "Inputs and outputs must have same number of rows"
    assert len(vd_xs) == len(vd_ys), "Inputs and outputs must have same number of rows"

    def run():
        l = OrderedRuleLearner()
        l.fit(td_xs, td_ys)
        return (l.rules, l.predict(vd_xs))

    rules, pred_ys = timeit(run, "Learn Rules from data")
    rec, prec, f1, acc = rpf1a(vd_ys, pred_ys, 1)
    print "Recall %f Precision %f F1 %f Accuracy: %f\n" % (rec, prec, f1, acc)
    for r in rules:
        print r
    pass