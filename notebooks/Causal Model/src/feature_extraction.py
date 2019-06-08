from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Set, Union

from build_chains import build_chains, extend_chains, get_distinct_chains
from causal_model_features import build_cb_causal_model, build_sc_causal_model, distance_between
from parse_generator import get_all_combos, get_max_probs
from parser_inputs import ParserInputs, ParserInputsEssayLevel
from sample_parses import get_top_parses

import numpy as np

# construct Causal Models
CM_MDL = build_cb_causal_model()
SC_MDL = build_sc_causal_model()

TERMINAL = "50"
cb_mdl = build_cb_causal_model()
sc_mdl = build_sc_causal_model()


def to_freq_feats(feats, freq_feats):
    new_feats = defaultdict(float)
    for f, v in feats.items():
        if f in freq_feats:
            new_feats[f] = v
    return new_feats

def filter_by_min_freq(xs: List[Union[ParserInputs, ParserInputsEssayLevel]],
                       feat_freq: Dict[str, int], min_freq: int)->List[Union[ParserInputs, ParserInputsEssayLevel]]:
    if min_freq <= 1:
        return xs
    freq_feats = set((f for f, cnt in feat_freq.items() if cnt >= min_freq))
    for parser_input in xs:
        parser_input.opt_features = to_freq_feats(parser_input.opt_features, freq_feats)
        parser_input.other_features_array = [to_freq_feats(x, freq_feats)
                                             for x in parser_input.other_features_array]
    return xs

def to_short_tag(tag: str)->str:
    return tag.replace("Causer:","").replace("Result:", "")

def extract_features_from_parse(parse: Tuple[str], crel2probs: Dict[str, List[float]], causal_model: Dict[str,str])->Dict[str,float]:
    feats = defaultdict(float)
    tree = defaultdict(set)  # maps causers to effects for building chains
    max_probs = []
    code_tally = defaultdict(float)

    num_crels = len(parse)
    pairs = set()
    inverted_count = 0
    num_fwd = 0
    num_equal = 0

    # differences inbetween codes
    num_adjacent = 0
    num_crossing_crels = 0 # crels crossing chains - not reachable
    abs_diffs = []
    distinct_codes = set()

    for crel in parse:
        probs = crel2probs[crel]
        max_p = max(probs)
        max_probs.append(max_p)
        feats["CREL_{crel}-MAX(prob)".format(crel=crel)] = max_p
        feats["CREL_{crel}-MIN(prob)".format(crel=crel)] = min(probs)
        feats["CREL_{crel}-pred-count".format(crel=crel)] = len(probs)
        feats["CREL_{crel}-pred-count={count}".format(crel=crel, count=len(probs))] = 1

        # with type - Causer or Effect
        l, r = crel.split("->")
        code_tally["Tally-" + l] += 1
        code_tally["Tally-" + r] += 1

        # without type
        l_short, r_short = to_short_tag(l), to_short_tag(r)
        distinct_codes.add(l_short)
        distinct_codes.add(r_short)

        # numeric difference between codes
        abs_difference = distance_between(l_short, r_short, causal_model)
        if abs_difference == -1:
            num_crossing_crels += 1
        else:
            abs_diffs.append(abs_difference)

            if abs_difference == 1:
                num_adjacent += 1

            # Forward relations
            if l_num < r_num:
                num_fwd += 1

        # Equal to
        if l_short == r_short:
            num_equal += 1

        code_tally["Tally-" + l_short] += 1
        code_tally["Tally-" + r_short] += 1
        # ordering of the codes, ignoring the causal direction
        feats["CREL_" + l_short + ":" + r_short] = 1

        # build tree structure so we can retrieve the chains
        tree[l_short].add(r_short)

        # track whether the rule exists in the opposite direction
        pairs.add((l_short, r_short))
        if (r_short, l_short) in pairs:
            inverted_count += 1

    if inverted_count:
        feats["Inv-inverted"] = 1
        feats["Inv-num_inverted"] = inverted_count
        feats["Inv-propn_inverted"] = inverted_count / num_crels
    else:
        feats["Inv-not_inverted"] = 1

    # Propn feats
    PROPN = "Propn_"
    DIFF  = "Diff_"
    if num_crels > 0:
        feats[PROPN + "fwd"] = num_fwd / num_crels
        feats[PROPN + "equal"] = num_equal / num_crels
        feats[PROPN + "unique_codes"] = len(distinct_codes) / (2*num_crels) # 2 codes per crel

        feats[DIFF + "adjacent_codes"] = num_adjacent / (num_crels)
        feats[DIFF + "MEAN_diff"]      = np.mean(abs_diffs)
        feats[DIFF + "MED_diff"]       = np.median(abs_diffs)
        feats[DIFF + "MAX_diff"]       = np.max(abs_diffs)

    # counts
    feats.update(code_tally)

    feats["num_crels"] = num_crels
    feats["num_crels=" + str(len(parse))] = 1  # includes a tag for the empty parse
    # for i in range(1, 11):
    #     if num_crels <= i:
    #         feats["num_crels<={i}".format(i=i)] = 1
    #     else:
    #         feats["num_crels>{i}".format(i=i)] = 1

    # combination of crels
    # need to sort so that order of a and b is consistent across parses
    pairs = combinations(sorted(parse), r=2)
    for a, b in pairs:
        feats["CREL_Pair-{a}|{b}".format(a=a, b=b)] = 1

    # chains
    chains = build_chains(tree)
    causer_chains = extend_chains(chains)
    max_chain_len = 0
    for ch in causer_chains:
        feats["CChain-:" + ch] = 1
        max_chain_len = max(max_chain_len, len(ch.split(",")))

    feats["CChainStats-MaxChain_Len=" + str(max_chain_len)] = 1
    distinct_chains = get_distinct_chains(chains)
    num_distinct = len(distinct_chains)
    feats["CChainStats-num_distinct_chains=" + str(num_distinct)] = 1
    # for i in range(6):
    #     if num_distinct <= i:
    #         feats["CChainStats-num_distinct_chains <=" + str(i)] = 1
    #     else:
    #         feats["CChainStats-num_distinct_chains > " + str(i)] = 1

    if num_distinct > 0:
        feats["CChainStats-crels_per_distinct_chain"] = num_crels / num_distinct

    if max_probs:  # might be an empty parse
        for cutoff in [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            above = len([p for p in max_probs if p >= cutoff])
            feats["Above-{cutoff}".format(cutoff=cutoff)] = above
            feats["Above-%-{cutoff}".format(cutoff=cutoff)] = above / len(max_probs)
            if above == len(max_probs):
                feats["Above-All-Above-{cutoff}".format(cutoff=cutoff)] = 1

        feats["Prob-avg-prob"] = np.mean(max_probs)
        feats["Prob-med-prob"] = np.median(max_probs)
        feats["Prob-prod-prob"] = np.product(max_probs)
        feats["Prob-min-prob"] = np.min(max_probs)
        feats["Prob-max-prob"] = np.max(max_probs)
        for p in [5, 10, 25, 75, 90, 95]:
            feats["Prob-{p}%-prob".format(p=p)] = np.percentile(max_probs, p)
        # geometric mean
        feats["Prob-geo-mean"] = np.prod(max_probs) ** (1 / len(max_probs))
    return feats

def get_crels_above(crel2maxprob: Dict[str, float], threshold: float)->List[str]:
    return [k for k, p in crel2maxprob.items() if p >= threshold]

def get_features_from_probabilities(essay2probs: Dict[str, Dict[str, List[float]]],
                                    name2crels: Dict[str, Set[str]], top_n: int,
                                    min_feat_freq:int =1, min_prob:float = 0.0)->List[ParserInputs]:
    xs = []
    feat_freq = defaultdict(int)

    for ename, crel2probs in essay2probs.items():

        act_crels = name2crels[ename]
        # used so we can do some sampling below to generate different parses
        crel2maxprob = get_max_probs(crel2probs)
        crel2probs = dict(crel2probs)

        keys = list(crel2probs.keys())
        n_parses = 2 ** len(keys)

        increment = 0.05
        threshold = min_prob - increment
        while n_parses > 2 * top_n and threshold < 1.0:
            threshold += increment
            keys = get_crels_above(crel2maxprob, threshold)
            n_parses = 2 ** len(keys)

        if n_parses > 2 * top_n:
            print("n_parses={n_parses} still exceeded max={max_p} at p={p:.4f}".format(
                p=threshold, n_parses=n_parses, max_p=top_n))
            parses = get_top_parses(crel2maxprob)
        else:
            parses = get_all_combos(keys)

        # constrain optimal parse to only those crels that are predicted
        # This is because often the parser won't produce the optimal parse (ground truth) in the set of generated parses
        # So here the target is to learn the best match from the set of generated parses
        opt_parse = tuple(sorted(act_crels.intersection(crel2probs.keys())))
        x = ParserInputs(essay_name=ename, opt_parse=opt_parse, all_parses=parses, crel2probs=crel2probs)
        xs.append(x)

        # Get unique features for essay
        all_feats = set()
        for fts in x.all_feats_array:
            all_feats.update(fts.keys())

        for ft in all_feats:
            feat_freq[ft] += 1

    assert len(xs) == len(essay2probs), "Parses for all essays should be generated"
    return filter_by_min_freq(xs, feat_freq, min_feat_freq)

def dict2parse(dct: Dict[str, List[float]])->Tuple[str]:
    return tuple(sorted(dct.keys())) # type: Tuple[str]

def get_features_essay_level(essay2parses: Dict[str, List[Dict[str, List[float]]]],
                             name2crels: Dict[str, Set[str]], min_feat_freq:int =1)->List[ParserInputsEssayLevel]:
    xs = []
    feat_freq = defaultdict(int)

    for ename, all_parse_dict in essay2parses.items():

        act_crels = name2crels[ename]           # type: Set[str]

        # for computing the optimal parse
        all_parses = []                         # type: List[Tuple[str]]

        min_cost = 999999
        opt_parse = None
        opt_parse_dict = None
        for parse_dict in all_parse_dict:       # type: Dict[str, List[float]]

            parse_tuple = dict2parse(parse_dict) # type: Tuple[str]
            all_parses.append(parse_tuple)
            set_parse = set(parse_tuple)         # type: Set[str]

            fp = set_parse - act_crels
            fn = act_crels - set_parse
            cost = len(fp) + len(fn)  # type: int
            if cost < min_cost:
                min_cost = cost
                opt_parse_dict = parse_dict
                opt_parse = parse_tuple

        assert opt_parse is not None

        #TODO - we want to include the cum prob from the parse action result as a feature
            # - how to compute this for the optimal parse?

        # constrain optimal parse to only those crels that are predicted
        # This is because often the parser won't produce the optimal parse (ground truth) in the set of generated parses
        # So here the target is to learn the best match from the set of generated parses

        x = ParserInputsEssayLevel(essay_name=ename, opt_parse=opt_parse, opt_parse_dict=opt_parse_dict,
                                   all_parses=all_parses, all_parses_dict=all_parse_dict)
        xs.append(x)

        # Get unique features for essay
        all_feats = set()
        for fts in x.all_feats_array:
            all_feats.update(fts.keys())

        for ft in all_feats:
            feat_freq[ft] += 1

    assert len(xs) == len(essay2parses), "Parses for all essays should be generated"
    return filter_by_min_freq(xs, feat_freq, min_feat_freq)
