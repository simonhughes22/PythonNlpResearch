from collections import defaultdict
from typing import Tuple, List, Dict

from causal_model_features import build_cb_causal_model, build_sc_causal_model, CausalModelType
from costs import compute_costs

# construct Causal Models
CM_MDL = build_cb_causal_model()
SC_MDL = build_sc_causal_model()

def copy_dflt_dict(d: Dict[str, float])->Dict[str, float]:
    copy = defaultdict(d.default_factory)
    copy.update(d)
    return copy

class ParserInputs(object):
    def __init__(self, essay_name, opt_parse, all_parses, crel2probs,  causal_model_type, compute_feats=True):
        # placing here as a cyclic dependency with feature_extraction file, so loading only at run time
        from feature_extraction import extract_features_from_parse
        causal_model = CM_MDL if causal_model_type == CausalModelType.CORAL_BLEACHING else SC_MDL
        self.causal_model_type = causal_model_type

        self.essay_name = essay_name
        self.opt_parse = opt_parse
        self.crel2probs = crel2probs

        if compute_feats:
            self.opt_features = extract_features_from_parse(opt_parse, crel2probs)

            other_parses = []
            other_feats_array = []
            all_feats_array = []
            for p in all_parses:
                feats = extract_features_from_parse(p, crel2probs, causal_model)
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
                         all_parses=self.all_parses, crel2probs=self.crel2probs,
                         causal_model_type=self.causal_model_type,
                         compute_feats=False)

        c.other_parses = list(self.other_parses)
        c.other_costs_array = list(self.other_costs_array)
        return c

    def clone(self):
        c = ParserInputs(essay_name=self.essay_name, opt_parse=self.opt_parse,
                         all_parses=self.all_parses, crel2probs=self.crel2probs,
                         causal_model_type=self.causal_model_type,
                         compute_feats=False)

        c.all_feats_array = [copy_dflt_dict(f) for f in self.all_feats_array]
        c.opt_features = copy_dflt_dict(self.opt_features)
        c.other_parses = list(self.other_parses)
        c.other_features_array = [copy_dflt_dict(f) for f in self.other_features_array]
        c.other_costs_array = list(self.other_costs_array)
        return c

class ParserInputsEssayLevel(object):
    def __init__(self, essay_name: str,
                 opt_parse: Tuple[str], opt_parse_dict: Dict[str, List[float]],
                 all_parses: List[Tuple[str]], all_parses_dict: List[Dict[str, List[float]]],
                 causal_model_type: str, compute_feats : bool = True):

        # placing here as a cyclic dependency with feature_extraction file, so loading only at run time
        from feature_extraction import extract_features_from_parse
        causal_model = CM_MDL if causal_model_type == CausalModelType.CORAL_BLEACHING else SC_MDL
        self.causal_model_type = causal_model_type

        self.essay_name = essay_name
        self.opt_parse  = opt_parse
        self.all_parses = all_parses

        assert len(all_parses) == len(all_parses_dict)

        # needed for cloning
        self.__opt_parse_dict__ = opt_parse_dict
        self.__all_parses_dict__ = list(all_parses_dict)  # type: List[Dict[str, List[float]]]

        if compute_feats:

            other_parses = []                           # type: List[Tuple[str]]
            other_feats_array = []                      # type: List[Dict[str,float]]
            all_feats_array = []                        # type: List[Dict[str,float]]

            for parse_tuple, parse_dict in zip(all_parses, self.__all_parses_dict__):

                feats = extract_features_from_parse(parse_tuple, parse_dict, causal_model) # type: Dict[str,float]
                all_feats_array.append(feats)

                # Don't store optimal parse in other parses, otherwise it will be ranked against itself
                if parse_tuple != self.opt_parse:
                    other_parses.append(parse_tuple)
                    other_feats_array.append(feats)

            # Public
            self.opt_features = extract_features_from_parse(self.opt_parse, opt_parse_dict, causal_model)

            # Public
            self.all_feats_array = all_feats_array
            self.other_parses = other_parses

            # Public
            self.other_features_array = other_feats_array
            # Public
            self.other_costs_array = compute_costs(self)

    def clone_without_feats(self):
        c = ParserInputsEssayLevel(essay_name=self.essay_name,
                                   opt_parse=self.opt_parse, opt_parse_dict=self.__opt_parse_dict__,
                                   all_parses=self.all_parses, all_parses_dict=self.__all_parses_dict__,
                                   causal_model_type=self.causal_model_type,
                                   compute_feats=False)

        c.other_parses = list(self.other_parses)
        c.other_costs_array = list(self.other_costs_array)
        return c

    def clone(self):
        c = ParserInputsEssayLevel(essay_name=self.essay_name,
                                   opt_parse=self.opt_parse, opt_parse_dict=self.__opt_parse_dict__,
                                   all_parses=self.all_parses, all_parses_dict=self.__all_parses_dict__,
                                   causal_model_type=self.causal_model_type,
                                   compute_feats=False)

        c.opt_features = copy_dflt_dict(self.opt_features)
        c.all_feats_array = [copy_dflt_dict(f) for f in self.all_feats_array]
        c.other_features_array = [copy_dflt_dict(f) for f in self.other_features_array]

        c.other_parses = list(self.other_parses)
        c.other_costs_array = list(self.other_costs_array)
        return c