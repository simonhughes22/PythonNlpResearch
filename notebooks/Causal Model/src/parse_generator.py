from collections import defaultdict
from itertools import combinations

from searn_parser_breadth_first import geo_mean

def get_possible_crels(predicted_tags):
    if len(predicted_tags) < 2:
        return set()
    predicted_tags = sorted(predicted_tags)
    pred_crels = set()
    for a,b in combinations(predicted_tags, 2):
        pred_crels.add("Causer:{a}->Result:{b}".format(a=a, b=b))
        pred_crels.add("Causer:{b}->Result:{a}".format(a=a, b=b))
    return pred_crels

def to_canonical_parse(crels):
    return tuple(sorted(crels))

def collapse_sent_parse(pred_parses):
    crel2prob = defaultdict(list)
    for pact in pred_parses:
        act_seq = pact.get_action_sequence()
        for act in act_seq:
            if not act.relations:
                continue

            assert act.lr_action_prob >= 0
            prob = geo_mean([act.action_prob * act.lr_action_prob])
            for r in act.relations:
                crel2prob[r].append(prob)
    return crel2prob

def merge_crel_probs(a, b):
    for k,v in b.items():
        a[k].extend(v)
    return a

def get_max_probs(crel2probs):
    crel2max_prob = dict()
    for crel, probs in crel2probs.items():
        crel2max_prob[crel] = max(probs)
    return crel2max_prob

def get_all_combos(items):
    # enforces a consistent ordering for the resulting tuples
    items = sorted(items)
    cbos = [()] # seed with the empty combo
    for i in range(1, len(items)+1):
        cbos.extend(combinations(items,i))
    return cbos

