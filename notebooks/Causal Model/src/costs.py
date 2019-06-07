from typing import List

def compute_costs(parser_input)->List[int]:
    opt_parse = parser_input.opt_parse
    other_parses = parser_input.other_parses

    other_costs = [] # type: List[int]
    op = set(opt_parse)
    for p in other_parses:
        p = set(p)
        fp = p - op
        fn = op - p
        cost = len(fp) + len(fn) # type: int
        other_costs.append(cost)
    return other_costs # type: List[int]