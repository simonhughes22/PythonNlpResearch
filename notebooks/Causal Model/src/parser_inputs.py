from collections import defaultdict
from typing import Tuple, List, Dict

from costs import compute_costs

def copy_dflt_dict(d: Dict[str, float])->Dict[str, float]:
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

def dict2parse(dct: Dict[str, List[float]])->Tuple[str]:
    return tuple(sorted(dct.keys())) # type: Tuple[str]

class ParserInputsEssayLevel(object):
    def __init__(self, essay_name: str, opt_parse_dict: Dict[str, List[float]],
                 all_parse_dict: List[Dict[str, List[float]]], compute_feats : bool = True):

        # placing here as a cyclic dependency with feature_extraction file, so loading only at run time
        from feature_extraction import extract_features_from_parse

        self.essay_name = essay_name
        self.opt_parse  = dict2parse(opt_parse_dict)    # type: Tuple[str]

        # needed for cloning
        self.__opt_parse_dict__ = opt_parse_dict
        self.__all_parse_dict__ = list(all_parse_dict)  # type: List[Dict[str, List[float]]]

        all_parses = []                                 # type: List[Tuple[str]]
        if compute_feats:

            other_parses = []                           # type: List[Tuple[str]]
            other_feats_array = []                      # type: List[Dict[str,float]]
            all_feats_array = []                        # type: List[Dict[str,float]]

            for parse_dict in self.__all_parse_dict__:

                parse_tuple = dict2parse(parse_dict)    # type: Tuple[str]
                all_parses.append(parse_tuple)

                feats = extract_features_from_parse(parse_tuple, parse_dict) # type: Dict[str,float]
                all_feats_array.append(feats)

                # Don't store optimal parse in other parses, otherwise it will be ranked against itself
                if parse_tuple != self.opt_parse:
                    other_parses.append(parse_tuple)
                    other_feats_array.append(feats)

            # Public
            self.opt_features = extract_features_from_parse(self.opt_parse, opt_parse_dict)

            # Public
            self.all_feats_array = all_feats_array
            self.other_parses = other_parses

            # Public
            self.other_features_array = other_feats_array
            # Public
            self.other_costs_array = compute_costs(self)

        self.all_parses = all_parses

    def clone_without_feats(self):
        c = ParserInputsEssayLevel(essay_name=self.essay_name, opt_parse_dict=self.__opt_parse_dict__,
                                   all_parse_dict=self.__all_parse_dict__, compute_feats=False)

        c.other_parses = list(self.other_parses)
        c.other_costs_array = list(self.other_costs_array)
        return c

    def clone(self):
        c = ParserInputsEssayLevel(essay_name=self.essay_name, opt_parse_dict=self.__opt_parse_dict__,
                                   all_parse_dict=self.__all_parse_dict__, compute_feats=False)

        c.all_feats_array = [copy_dflt_dict(f) for f in self.all_feats_array]
        c.opt_features = copy_dflt_dict(self.opt_features)

        c.other_parses = list(self.other_parses)
        c.other_features_array = [copy_dflt_dict(f) for f in self.other_features_array]
        c.other_costs_array = list(self.other_costs_array)
        return c