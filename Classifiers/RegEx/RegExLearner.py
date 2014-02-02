__author__ = 'simon.hughes'

from itertools import imap, izip
from Metrics import recall, precision, f1_score, rpf1
from DocumentFrequency import compute_document_frequency
from Rule import Rule, DisjointRule, PositiveNotNegativeRule

class RegExLearner(object):
    def __init__(self, rule_score_fn, global_score_fn, min_pct_coverage):
        """
            single_pattern_fn : fn used to evaluate which single new patterns to add ([] expected, [] actual): float
            full_pattern_fn :   fn used to evaluate the entire set of rules ([] expected, [] actual): float
            min_pct_coverage :  min % positive examples matched by any rule : 0 - 100
        """
        self.rule_score_fn, self.global_score_fn = rule_score_fn, global_score_fn
        self.min_pct_coverage = float(min_pct_coverage)
        self.positive_rules = []
        self.negative_rules = []

    """ Public """

    def __above_doc_freq__(self, doc_freq, minimum):
        return set(k for k, v in doc_freq.items() if v >= minimum)

    def __add_negative_rules__(self, pos_comp_rule, current_best_global_score, xs, ys):

        print("Adding new negative rules")
        neg_rules = []
        best_global_score = current_best_global_score

        # include both sets of tokens to build negative rules
        tokens = self.negative_tokens.union(self.positive_tokens)
        neg_ys = map(lambda y: self.positive_label if y == 0 else 0, ys)

        # While we keep improving the global score (do while pattern)
        """ Add new negative rules """
        while True:
            # get next neg rule
            previous_best_pattern = []
            current_best_pattern = []
            best_rule_score = 0
            # Only set this if the global score increases

            improved_local_score = True
            # Grow rule while each iteration improves on the previous (adding to the length)
            candidates = []
            """ Grow new rule """
            while improved_local_score:
                improved_local_score = False
                previous_best_pattern = current_best_pattern[:]

                insertion_indices = range(len(previous_best_pattern) + 1)
                for token in tokens:
                    for insertion_index in insertion_indices:
                        new_pattern = previous_best_pattern[:insertion_index] + [token] + previous_best_pattern[insertion_index:]

                        new_rule = Rule(new_pattern)
                        neg_predictions = self.__predict_labels_from_rule__(new_rule, xs)
                        new_rule_score = self.rule_score_fn(neg_ys, neg_predictions)

                        if new_rule_score > best_rule_score:
                            current_best_pattern = new_pattern
                            best_rule_score = new_rule_score
                            improved_local_score = True
                            candidates.append(new_rule)

            best_new_rule = None
            tmp_best_global = best_global_score
            # Did we improve global score?
            for candidate in candidates:
                tmp_negative_rules = neg_rules + [candidate]
                new_combo_rule = PositiveNotNegativeRule(pos_comp_rule, DisjointRule(tmp_negative_rules))
                predictions = self.__predict_labels_from_rule__(new_combo_rule, xs)
                new_global_score = self.global_score_fn(ys, predictions)

                if new_global_score > tmp_best_global:
                    tmp_best_global = new_global_score
                    # has to improve the global score to be considered
                    best_new_rule = candidate

            if best_new_rule is not None:
                neg_rules.append(best_new_rule)
                best_global_score = tmp_best_global
                print "New best global score (from adding negative rules: ", str(round(best_global_score, 4)), "for", best_new_rule
            else:
                break

        """
            Current problem with this is that it typically needs to grow a
            long rule in order to improve performance
        """

        # remove matched examples
        return neg_rules

    def __build_rules__(self, xs, ys):
        self.positive_rules = []

        instances = zip(xs, ys)

        work_list = instances[:]
        tokens = self.positive_tokens.copy()

        composite_pos_rule = None
        max_global_score = float('-inf')

        print("Iteratively adding rules")
        while len(work_list) > self.min_positive_rules_covered:
            next_rule, work_list, tokens = self.__get_next_rule__(tokens, work_list)
            if next_rule is None:
                print "Quitting iterative rule growth: failed to generate a new rule that matched sufficient examples"
                break

            #Compute score from global score
            tmp_rules = self.positive_rules[:]
            tmp_rules.append(next_rule)

            new_composite_rule = DisjointRule(tmp_rules)
            predictions = self.__predict_labels_from_rule__(new_composite_rule, xs)
            score = self.global_score_fn(ys, predictions)

            if score <= max_global_score:
                print "Quitting iterative rule growth: Max global score failed to improve"
                break
            print "New best global score: ", str(round(score, 4)), "for", next_rule
            max_global_score = score
            # Only add a rule if it improves the global score
            self.positive_rules.append(next_rule)
            composite_pos_rule = new_composite_rule

        #self.__improve_global_score__(xs, ys, max_global_score)
        self.negative_rules = self.__add_negative_rules__(composite_pos_rule, max_global_score, xs, ys)

        #matched_docs, matched_labels, unmatched_docs, unmatched_labels \
        #   = self.__partition_by_regex_classification__(composite_pos_rule, xs, ys)

        pass

    def __get_next_rule__(self, tokens, instances):

        best_pattern = []
        best_match_count = -1

        best_score = float('-inf')
        un_matched_instances = []

        _, ys = zip(*instances)

        improved = True
        while improved:
            improved = False
            current_best_pattern = best_pattern[:]
            insertion_indices = range(len(current_best_pattern) + 1)

            for token in tokens:
                for insertion_index in insertion_indices:
                    new_pattern = best_pattern[:insertion_index] + [token] + best_pattern[insertion_index:]

                    new_rule = Rule(new_pattern)
                    un_matched = []
                    predictions = []
                    match_count = 0
                    for x, y in instances:
                        if new_rule.matches(x):
                            match_count += 1
                            predictions.append(self.positive_label)
                        else:
                            predictions.append(0)
                            un_matched.append((x, y))

                    score = self.rule_score_fn(ys, predictions)
                    if score >= best_score and match_count >= self.min_positive_rules_covered:

                        # If tied, always prefer ones that match more instances
                        if score == best_score and match_count <= best_match_count:
                            continue

                        current_best_pattern = new_pattern
                        best_match_count = match_count

                        best_score = score
                        improved = True
                        un_matched_instances = un_matched
                    pass # End for
                pass # End for
            best_pattern = current_best_pattern
            pass

        if len(best_pattern) == 0:
            return None, None, None

        best_rule = Rule(best_pattern)

        print "\tNew rule added: {0}\n\t\tRule Score: {1}\n\t\tMatches: {2}".format(best_rule, best_score,
                                                                                    best_match_count)

        """ Compute remaining tokens """
        un_matched_positives, un_matched_negatives = self.__partition_by_class__(un_matched_instances)
        positive_doc_freq = compute_document_frequency(un_matched_positives)
        remaining_tokens = self.__above_doc_freq__(positive_doc_freq, self.min_positive_rules_covered)

        return best_rule, un_matched_instances, remaining_tokens

    def __improve_global_score__(self, xs, ys, current_score):

        print("Improving global score")
        best_score = current_score

        # While we keep improving
        while True:
            # Create new pattern
            best_pattern = []
            # Eval until upto 3 tokens
            improved = True
            while improved:
                improved = False
                insertion_indices = range(len(best_pattern) + 1)
                current_best_pattern = best_pattern[:]
                for token in self.positive_tokens:
                    for insertion_index in insertion_indices:
                        new_pattern = best_pattern[:insertion_index] + [token] + best_pattern[insertion_index:]
                        new_rule = Rule(new_pattern)
                        tmp_rules = self.positive_rules + [new_rule]

                        predictions = self.__predict_labels_from_rule__(tmp_rules, xs)
                        new_score = self.global_score_fn(ys, predictions)
                        if new_score > best_score:
                            improved = True
                            best_score = new_score
                            current_best_pattern = new_pattern
                            # No improvement, we want to stop here as this is a greedy algorithm
                            # technically we could search deeper and possibly improve
                best_pattern = current_best_pattern
            if len(best_pattern) == 0:
                return
            else:
                new_rule = Rule(best_pattern)
                self.positive_rules.append(new_rule)
                print("New best global score: " + str(round(best_score, 4))) + " for " + str(new_rule)
        pass # end improve global score

    def __partition_by_class__(self, instances):
        positive_tuples = filter(lambda (x, y): y == self.positive_label, instances)
        negative_tuples = filter(lambda (x, y): y != self.positive_label, instances)
        positive_docs = zip(*positive_tuples)[0] if len(positive_tuples) > 0 else []
        negative_docs = zip(*negative_tuples)[0] if len(negative_tuples) > 0 else []
        return positive_docs, negative_docs

    def __partition_by_regex_classification__(self, comp_rule, xs, ys):
        matched_docs, matched_labels, unmatched_docs, unmatched_labels = [], [], [], []
        for x, y in izip(xs, ys):
            if comp_rule.matches(x):
                matched_docs.append(x)
                matched_labels.append(y)
            else:
                unmatched_docs.append(x)
                unmatched_labels.append(y)
        return matched_docs, matched_labels, unmatched_docs, unmatched_labels

    def __predict_labels_from_rule__(self, rule, xs):
        predictions = map(lambda x: self.positive_label if rule.matches(x) else 0, xs)
        return predictions

    def __repr__(self):
        s = "Positive Rules:\n\t"
        for rstr in sorted(imap(lambda r: str(r.tokens), self.positive_rules)):
            s += rstr + "\n\t"
        s = s[:-1]
        s += "\nNegative Rules:\n\t"
        for rstr in sorted(imap(lambda r: str(r.tokens), self.negative_rules)):
            s += rstr + "\n\t"
        return s

    def fit(self, xs, ys):

        self.positive_label = max(ys)

        # Split into positive and negative docs
        positive_docs, negative_docs = self.__partition_by_class__(zip(xs, ys))

        self.min_positive_rules_covered = self.min_pct_coverage * 0.01 * len(positive_docs)
        self.min_negative_rules_covered = self.min_pct_coverage * 0.01 * len(negative_docs)

        pos_doc_freq = compute_document_frequency(positive_docs)
        neg_doc_freq = compute_document_frequency(negative_docs)

        self.positive_tokens = self.__above_doc_freq__(pos_doc_freq, self.min_positive_rules_covered)
        self.negative_tokens = self.__above_doc_freq__(neg_doc_freq, self.min_negative_rules_covered)

        self.__build_rules__(xs, ys)
        pass

    def predict(self, xs):
        p_rule = DisjointRule(self.positive_rules)
        n_rule = DisjointRule(self.negative_rules)
        return self.__predict_labels_from_rule__(PositiveNotNegativeRule(p_rule, n_rule), xs)

    pass

    """ END Public """


if __name__ == "__main__":

    def test_rule():
        pattern = ["a", "b", "c"]
        rule = Rule(pattern)

        w1 = "a fat b on a c"
        w2 = "xxx a b c"
        w3 = "a b c 000000"
        w4 = "a jjjj b c"
        w5 = "a b kkkk c"
        w6 = "xxx a xxx b xxx c xxx"
        w7 = "a b c"

        should_match = [w1, w2, w3, w4, w5, w6, w7]
        assert all(map(rule.matches, should_match))

        wa = "a b"
        wb = "b c"
        wc = "c"
        wx = "asd kjlsa"
        should_NOT_match = [wa, wb, wc, wx]

        assert not any(map(rule.matches, should_NOT_match))

    def get_positive(xs, ys):
        tuples = zip(xs, ys)
        return [x for (x, y) in tuples if y > 0]

    def print_positives(xs, ys):
        positives = get_positive(xs, ys)
        print("Positive sentences(" + str(len(positives)) + "): ")
        for p in positives:
            print("\t" + str(p))
        print ""

    def test_learner():

        instances = [
            (["a", "b", "c", "d"], 1),
            (["a", "b", "d", "c"], 1),
            (["c", "a", "b"], 1),
            (["a", "b"], 1),

            (["a", "b", "e"], 0),
            (["c", "b", "a"], 0),
            (["a", "c"], 0),
            (["a", "c"], 0),
            (["b", "c"], 0),
            (["b", "d"], 0),
            (["d"], 0),

        ]

        xs, ys = zip(*instances)

        learner = RegExLearner(precision, f1_score, 2.0)
        learner.fit(xs, ys)
        pred = learner.predict(xs)

        print_positives(xs, ys)
        r, p, f1 = rpf1(ys, pred)
        print "TD:\n\tRecall: {0}\n\tPrecision: {1}\n\tF1: {2}\n".format(r, p, f1)
        print str(learner)
        pass

    def test_learner_on_data():
        import GwData
        import WordTokenizer
        import numpy as np

        MINIMUM_COVERAGE_PCT = 2.0
        code = "53"
        print "Learning rules for code: " + code
        # '%%' is how you print a '%' in python given that it is a special char
        print "Mininum coverage: %d%%\n" % (MINIMUM_COVERAGE_PCT)

        data = GwData.GwData()
        xs = WordTokenizer.tokenize(data.documents, stem=False,  spelling_correct=False, remove_stop_words=False, min_word_count=1)

        ys = data.labels_for(code)

        def rule_score_fn(act_ys, predicted):
            r, p, f1 = rpf1(act_ys, predicted)
            return r * (p ** 0.5)

        shuffled_ixs = np.array(range(len(xs)))
        np.random.shuffle(shuffled_ixs)

        shuffled_xs = np.array(xs)[shuffled_ixs]
        shuffled_ys = np.array(ys)[shuffled_ixs]

        td_size = int(len(xs) * 0.9)

        td_xs, td_ys = shuffled_xs[0:td_size], shuffled_ys[0:td_size]
        vd_xs, vd_ys = shuffled_xs[td_size:], shuffled_ys[td_size:]

        assert len(td_xs) + len(vd_xs) == len(xs), "|TD| + |VD| == |D|"

        learner = RegExLearner(precision, f1_score, MINIMUM_COVERAGE_PCT)
        learner.fit(td_xs, td_ys)

        print_positives(xs, ys)
        print str(learner)

        # TD Performance
        td_pred = learner.predict(td_xs)
        r, p, f1 = rpf1(td_ys, td_pred)
        print "TD:\n\tRecall: {0}\n\tPrecision: {1}\n\tF1: {2}\n".format(r, p, f1)

        # VD performance
        vd_pred = learner.predict(vd_xs)
        r, p, f1 = rpf1(vd_ys, vd_pred)
        print "VD:\n\tRecall: {0}\n\tPrecision: {1}\n\tF1: {2}\n".format(r, p, f1)

        pass

#Run Suite
#test_rule()
#test_learner()
test_learner_on_data()

""" TODO
    1. Add negative rules
    2. Try removing words to generalize better
        Either do on final pattern and evaluate to see if F1 improves (by improving recall - quite likely).
            If so, may be an iterative process of simplification, then trying to add more rules using precision
        Or do on a separate held out test set, separate from VD set (like post pruning)
    3. Try something more akin to the algoritm described by pedro domingos:
        Learn positive and negative rules separately
        Pick rules using accuracy or IG * coverage
        Use voting when classifying sentence as positive or negative
    4. Merge rules: merge ruls where words overlap , e.g.
            global      temperatures    increase
            earth       temperatures    rise
            planet's    temperatures    risen

            => (global|earth|planet's) temperatures (rise|risen|increase)
            The resulting 'merged' rule is more general and will meet more cases
"""