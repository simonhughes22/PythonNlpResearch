from collections import defaultdict
from typing import Tuple, List, Dict

from costs import compute_costs

def copy_dflt_dict(d):
    copy = defaultdict(d.default_factory)
    copy.update(d)
    return copy

class ParserInputs(object):
    def __init__(self, essay_name, opt_parse, all_parses, crel2probs, compute_feats=True):
        # placing here as a cyclic dependency with feature_extraction file, so loading only at run time
        from feature_extraction import extract_features_from_parse

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

def dict2parse(dct):
    return tuple(sorted(dct.keys()))

class ParserInputsEssayLevel(object):
    def __init__(self, essay_name: str, opt_parse_dict: Dict[str, List[float]],
                 all_parses: List[Dict[str, List[float]]], compute_feats : bool = True):

        # placing here as a cyclic dependency with feature_extraction file, so loading only at run time
        from feature_extraction import extract_features_from_parse

        self.essay_name = essay_name
        self.__opt_parse__ = dict2parse(opt_parse_dict)
        self.__dict_parses__ = list(all_parses)

        if compute_feats:
            # Public
            self.opt_features = extract_features_from_parse(self.__opt_parse__, opt_parse_dict)

            other_parses = []
            other_feats_array = []
            all_feats_array = []
            for p in all_parses:
                feats = extract_features_from_parse(dict2parse(p), p)
                all_feats_array.append(feats)
                if p != opt_parse_dict:
                    other_parses.append(p)
                    other_feats_array.append(feats)

            self.__all_feats_array__ = all_feats_array
            self.__other_parses__ = other_parses

            # Public
            self.other_features_array = other_feats_array
            # Public
            self.other_costs_array = compute_costs(self)

        self.all_parses = all_parses

    def clone_without_feats(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.__opt_parse__,
                         all_parses=self.all_parses, crel2probs=self.__dict_parses__, compute_feats=False)

        c.other_parses = list(self.__other_parses__)
        c.other_costs_array = list(self.other_costs_array)
        return c

    def clone(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.__opt_parse__,
                         all_parses=self.all_parses, crel2probs=self.__dict_parses__, compute_feats=False)

        c.all_feats_array = [copy_dflt_dict(f) for f in self.__all_feats_array__]
        c.opt_features = copy_dflt_dict(self.opt_features)
        c.other_parses = list(self.__other_parses__)
        c.other_features_array = [copy_dflt_dict(f) for f in self.other_features_array]
        c.other_costs_array = list(self.other_costs_array)
        return c