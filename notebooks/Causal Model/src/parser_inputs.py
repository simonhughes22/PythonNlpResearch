from collections import defaultdict

from costs import compute_costs
from feature_extraction import extract_features_from_parse

def copy_dflt_dict(d):
    copy = defaultdict(d.default_factory)
    copy.update(d)
    return copy

class ParserInputs(object):
    def __init__(self, essay_name, opt_parse, all_parses, crel2probs, compute_feats=True):
        self.essay_name = essay_name
        self.opt_parse = opt_parse
        self.crel2probs = crel2probs

        if compute_feats:
            self.opt_features = extract_features_from_parse(opt_parse, crel2probs)

            other_parses = []
            other_feats_array = []
            all_feats_array = []
            for p in all_parses:
                feats = extract_features_from_parse(p, crel2probs)
                all_feats_array.append(feats)
                if p != opt_parse:
                    other_parses.append(p)
                    other_feats_array.append(feats)

            self.all_feats_array = all_feats_array
            self.other_parses = other_parses
            self.other_features_array = other_feats_array
            self.other_costs_array = compute_costs(self)

        self.all_parses = all_parses

    def clone_without_feats(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.opt_parse,
                         all_parses=self.all_parses, crel2probs=self.crel2probs, compute_feats=False)

        c.other_parses = list(self.other_parses)
        c.other_costs_array = list(self.other_costs_array)
        return c

    def clone(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.opt_parse,
                         all_parses=self.all_parses, crel2probs=self.crel2probs, compute_feats=False)

        c.all_feats_array = [copy_dflt_dict(f) for f in self.all_feats_array]
        c.opt_features = copy_dflt_dict(self.opt_features)
        c.other_parses = list(self.other_parses)
        c.other_features_array = [copy_dflt_dict(f) for f in self.other_features_array]
        c.other_costs_array = list(self.other_costs_array)
        return c

class ParserInputsEssayLevel(object):
    def __init__(self, essay_name, opt_parse, all_parses, crel2probs, compute_feats=True):
        self.essay_name = essay_name
        self.opt_parse = opt_parse
        self.crel2probs = crel2probs

        if compute_feats:
            self.opt_features = extract_features_from_parse(opt_parse, crel2probs)

            other_parses = []
            other_feats_array = []
            all_feats_array = []
            for p in all_parses:
                feats = extract_features_from_parse(p, crel2probs)
                all_feats_array.append(feats)
                if p != opt_parse:
                    other_parses.append(p)
                    other_feats_array.append(feats)

            self.all_feats_array = all_feats_array
            self.other_parses = other_parses
            self.other_features_array = other_feats_array
            self.other_costs_array = compute_costs(self)

        self.all_parses = all_parses

    def clone_without_feats(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.opt_parse,
                         all_parses=self.all_parses, crel2probs=self.crel2probs, compute_feats=False)

        c.other_parses = list(self.other_parses)
        c.other_costs_array = list(self.other_costs_array)
        return c

    def clone(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.opt_parse,
                         all_parses=self.all_parses, crel2probs=self.crel2probs, compute_feats=False)

        c.all_feats_array = [copy_dflt_dict(f) for f in self.all_feats_array]
        c.opt_features = copy_dflt_dict(self.opt_features)
        c.other_parses = list(self.other_parses)
        c.other_features_array = [copy_dflt_dict(f) for f in self.other_features_array]
        c.other_costs_array = list(self.other_costs_array)
        return c