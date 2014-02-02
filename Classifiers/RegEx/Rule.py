__author__ = 'simon.hughes'

import re

class RuleABC(object):
    def matches(self, tokens):
        raise NotImplementedError("Implement matches in the concrete derived class")

class Rule(RuleABC):
    WC = "([a-z0-9])+"
    START = "(" + WC + "\s)*"
    MID = "\s(" + WC + "\s)*"
    END = "(\s" + WC + ")*"

    def __init__(self, tokens):
        self.tokens = tokens
        self.reg_ex_str = Rule.START + Rule.MID.join(tokens) + Rule.END
        self.reg_ex = re.compile(self.reg_ex_str)

    def matches(self, tokens):
        stringy = " ".join(tokens)
        return self.reg_ex.match(stringy) is not None

    def __repr__(self):
        return str(self.tokens)

class DisjointRule(RuleABC):
    def __init__(self, rules):
        self.rules = rules

    def matches(self, tokens):
        #TODO order by number matched to improve speed
        for r in self.rules:
            if r.matches(tokens):
                return True
        return False

    def __repr__(self):
        s = ""
        for r in self.rules:
            s += str(r) + "\n"
        return s

class PositiveNotNegativeRule(RuleABC):
    def __init__(self, positive_rule, negative_rule):
        self.positive_rule, self.negative_rule = positive_rule, negative_rule

    def matches(self, tokens):
        return self.positive_rule.matches(tokens) and not self.negative_rule.matches(tokens)
