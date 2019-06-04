from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple

from build_chains import build_chains, extend_chains, get_distinct_chains
from parse_generator import get_all_combos, get_max_probs
from parser_inputs import ParserInputs, ParserInputsEssayLevel
from sample_parses import get_top_parses

import numpy as np

def to_freq_feats(feats, freq_feats):
    new_feats = defaultdict(float)
    for f, v in feats.items():
        if f in freq_feats:
            new_feats[f] = v
    return new_feats

def filter_by_min_freq(xs, feat_freq, min_freq):
    if min_freq <= 1:
        return xs
    freq_feats = set((f for f, cnt in feat_freq.items() if cnt >= min_freq))
    for parser_input in xs:
        parser_input.opt_features = to_freq_feats(parser_input.opt_features, freq_feats)
        parser_input.other_features_array = [to_freq_feats(x, freq_feats)
                                             for x in parser_input.other_features_array]
    return xs

def to_short_tag(tag):
    return tag.replace("Causer:","").replace("Result:", "")

def extract_features_from_parse(parse: Tuple[str], crel2probs: Dict[str, List[float]] ):
    feats = defaultdict(float)
    tree = defaultdict(set)  # maps causers to effects for building chains
    max_probs = []
    code_tally = defaultdict(float)

    pairs = set()
    inverted_count = 0
    for crel in parse:
        probs = crel2probs[crel]
        max_p = max(probs)
        max_probs.append(max_p)
        feats["CREL_{crel}-MAX(prob)".format(crel=crel)] = max_p
        feats["CREL_{crel}-MIN(prob)".format(crel=crel)] = min(probs)
        feats["CREL_{crel}-pred-count".format(crel=crel)] = len(probs)
        feats["CREL_{crel}-pred-count={count}".format(crel=crel, count=len(probs))] = 1

        # with type
        l, r = crel.split("->")
        code_tally["Tally-" + l] += 1
        code_tally["Tally-" + r] += 1

        # without type
        l_short, r_short = to_short_tag(l), to_short_tag(r)
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
    else:
        feats["Inv-not_inverted"] = 1

    # counts
    feats.update(code_tally)
    num_crels = len(parse)
    feats["num_crels"] = num_crels
    feats["num_crels=" + str(len(parse))] = 1  # includes a tag for the empty parse
    for i in range(1, 11):
        if num_crels <= i:
            feats["num_crels<={i}".format(i=i)] = 1
        else:
            feats["num_crels>{i}".format(i=i)] = 1

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
    for i in range(6):
        if num_distinct <= i:
            feats["CChainStats-num_distinct_chains <=" + str(i)] = 1
        else:
            feats["CChainStats-num_distinct_chains > " + str(i)] = 1

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

def get_crels_above(crel2maxprob, threshold):
    return [k for k, p in crel2maxprob.items() if p >= threshold]

def get_features_from_probabilities(essay2probs, name2crels, top_n, min_feat_freq=1, min_prob=0.0):
    xs = []
    feat_freq = defaultdict(int)

    for ename, crel2probs in essay2probs.items():

        act_crels = name2crels[ename]
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

def get_features_essay_level(essay2parses, name2crels, min_feat_freq=1):
    xs = []
    feat_freq = defaultdict(int)

    def dict2parse(crel2maxprobs):
        return  [()] + [tuple(sorted(crel2maxprobs.keys()))]

    for ename, dict_parses in essay2parses.items():

        act_crels = name2crels[ename]
        all_predicted_crels = set()
        all_parses = []
        for p in dict_parses:
            crel2maxprob = get_max_probs(p)
            parse = dict2parse(crel2maxprob)
            all_parses.append(parse)
            all_predicted_crels.update(p.keys())

        #TODO - figure out the optimum parse probs to use
        #TODO - we want to include the cum prob from the parse action result as a feature
            # - how to compute this for the optimal parse?

        opt_parse = tuple(sorted(act_crels.intersection(all_predicted_crels.keys())))

        # constrain optimal parse to only those crels that are predicted
        # This is because often the parser won't produce the optimal parse (ground truth) in the set of generated parses
        # So here the target is to learn the best match from the set of generated parses

        x = ParserInputsEssayLevel(essay_name=ename, opt_parse_dict=opt_parse, all_parses=all_parses, dict_parses=dict_parses)
        xs.append(x)

        # Get unique features for essay
        all_feats = set()
        for fts in x.__all_feats_array__:
            all_feats.update(fts.keys())

        for ft in all_feats:
            feat_freq[ft] += 1

    assert len(xs) == len(essay2parses), "Parses for all essays should be generated"
    return filter_by_min_freq(xs, feat_freq, min_feat_freq)
