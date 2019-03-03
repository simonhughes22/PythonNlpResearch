from collections import defaultdict
from itertools import combinations

from build_chains import build_chains, extend_chains, get_distinct_chains
from parse_generator import get_all_combos, get_max_probs
from parser_inputs import ParserInputs
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

def extract_features_from_parse(parse, crel2probs):
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
        feats["{crel}-MAX(prob)".format(crel=crel)] = max_p
        feats["{crel}-MIN(prob)".format(crel=crel)] = min(probs)
        feats["{crel}-pred-count".format(crel=crel)] = len(probs)
        feats["{crel}-pred-count={count}".format(crel=crel, count=len(probs))] = 1

        # with type
        l, r = crel.split("->")
        code_tally[l] += 1
        code_tally[r] += 1

        # without type
        l_short, r_short = to_short_tag(l), to_short_tag(r)
        code_tally[l_short] += 1
        code_tally[r_short] += 1
        # ordering of the codes, ignoring the causal direction
        feats[l_short + ":" + r_short] = 1

        # build tree structure so we can retrieve the chains
        tree[l_short].add(r_short)

        # track whether the rule exists in the opposite direction
        pairs.add((l_short, r_short))
        if (r_short, l_short) in pairs:
            inverted_count += 1

    if inverted_count:
        feats["inverted"] = 1
        feats["num_inverted"] = inverted_count
    else:
        feats["not_inverted"] = 1

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
        feats["{a}|{b}".format(a=a, b=b)] = 1

    # chains
    chains = build_chains(tree)
    causer_chains = extend_chains(chains)
    max_chain_len = 0
    for ch in causer_chains:
        feats["CChain:" + ch] = 1
        max_chain_len = max(max_chain_len, len(ch.split(",")))
    feats["MaxChain_Len=" + str(max_chain_len)] = 1

    distinct_chains = get_distinct_chains(chains)
    num_distinct = len(distinct_chains)
    feats["num_distinct_chains=" + str(num_distinct)] = 1
    for i in range(6):
        if num_distinct <= i:
            feats["num_distinct_chains <=" + str(i)] = 1
        else:
            feats["num_distinct_chains > " + str(i)] = 1

    if num_distinct > 0:
        feats["crels_per_distinct_chain"] = num_crels / num_distinct

    if max_probs:  # might be an empty parse
        for cutoff in [0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
            above = len([p for p in max_probs if p >= cutoff])
            feats["Above-{cutoff}".format(cutoff=cutoff)] = above
            feats["%-Above-{cutoff}".format(cutoff=cutoff)] = above / len(max_probs)
            if above == len(max_probs):
                feats["All-Above-{cutoff}".format(cutoff=cutoff)] = 1

        feats["avg-prob"] = np.mean(max_probs)
        feats["med-prob"] = np.median(max_probs)
        feats["prod-prob"] = np.product(max_probs)
        feats["min-prob"] = np.min(max_probs)
        feats["max-prob"] = np.max(max_probs)
        for p in [5, 10, 25, 75, 90, 95]:
            feats["{p}%-prob".format(p=p)] = np.percentile(max_probs, p)
        # geometric mean
        feats["geo-mean"] = np.prod(max_probs) ** (1 / len(max_probs))
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
